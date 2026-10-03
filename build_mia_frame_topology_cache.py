"""Build paper-time synchronous topology samples from official MIA outputs.

Each sample is one synchronized frame ``t``.  Its nodes are exactly the
targets present in that frame, appearance is ROI-pooled from the frozen
official detector FPN at ``t``, and motion is the one-step state difference
from ``t-1`` when the same within-view track exists.  Geometry uses the exact
frozen MIA view-1-to-view-2 homography for that frame and never thresholds a
candidate by distance.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
import torch

from uav_tracking.mia_features import frame_features
from run_mia_frontend import build, detector_checkpoint
from uav_tracking.dataset import (
    discover_scene_pairs, list_frame_files, parse_cvat_tracks, select_scene_pairs,
)


def box_iou(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    if len(left) == 0 or len(right) == 0:
        return np.zeros((len(left), len(right)), dtype=np.float32)
    top_left = np.maximum(left[:, None, :2], right[None, :, :2])
    bottom_right = np.minimum(left[:, None, 2:], right[None, :, 2:])
    size = np.maximum(bottom_right - top_left, 0.0)
    intersection = size[..., 0] * size[..., 1]
    left_area = np.maximum(left[:, 2] - left[:, 0], 0.0) * np.maximum(left[:, 3] - left[:, 1], 0.0)
    right_area = np.maximum(right[:, 2] - right[:, 0], 0.0) * np.maximum(right[:, 3] - right[:, 1], 0.0)
    return intersection / np.maximum(left_area[:, None] + right_area[None, :] - intersection, 1e-8)


def match_truth(boxes: np.ndarray, records: list[dict], threshold: float) -> torch.Tensor:
    labels = torch.full((len(boxes),), -1, dtype=torch.long)
    if len(boxes) == 0 or not records:
        return labels
    truth_boxes = np.asarray([row["bbox"] for row in records], dtype=np.float32)
    overlap = box_iou(boxes, truth_boxes)
    rows, columns = linear_sum_assignment(1.0 - overlap)
    for row, column in zip(rows, columns):
        if overlap[row, column] >= threshold:
            labels[row] = int(records[column]["track_id"])
    return labels


def states(boxes: np.ndarray, width: float = 1920.0, height: float = 1080.0) -> torch.Tensor:
    if len(boxes) == 0:
        return torch.empty((0, 4), dtype=torch.float32)
    values = torch.as_tensor(boxes, dtype=torch.float32)
    return torch.stack([
        (values[:, 0] + values[:, 2]) * 0.5 / width,
        (values[:, 1] + values[:, 3]) * 0.5 / height,
        (values[:, 2] - values[:, 0]).clamp_min(1.0) / width,
        (values[:, 3] - values[:, 1]).clamp_min(1.0) / height,
    ], dim=1)


def mapped_box_reachable(boxes: np.ndarray, homography: np.ndarray,
                         width: float = 1920.0, height: float = 1080.0) -> np.ndarray:
    """Exact pre-distance MIA reachability check from ``common.py``."""
    if len(boxes) == 0:
        return np.zeros(0, dtype=bool)
    corners = boxes[:, [0, 1, 2, 3]].reshape(-1, 2, 2)
    mapped = cv2.perspectiveTransform(corners.astype(np.float32), homography)
    minimum = mapped.min(axis=1)
    maximum = mapped.max(axis=1)
    return ((minimum[:, 0] >= 0.0) & (minimum[:, 1] >= 0.0)
            & (maximum[:, 0] <= width) & (maximum[:, 1] <= height))


def frame_nodes(rows: list[list[float]], scene: str, view: int, frame: int,
                history: dict, previous_max_id: float | None = None,
                ) -> tuple[list[tuple], np.ndarray, torch.Tensor, torch.Tensor, torch.Tensor]:
    boxes = np.asarray([row[1:5] for row in rows], dtype=np.float32).reshape(-1, 4)
    spatial = states(boxes)
    trajectory = torch.zeros((len(rows), 5), dtype=torch.float32)
    is_new = torch.zeros(len(rows), dtype=torch.bool)
    occurrences = defaultdict(int)
    keys = []
    for index, row in enumerate(rows):
        track_id = int(row[0])
        occurrence = occurrences[track_id]
        occurrences[track_id] += 1
        token = (scene, int(view), track_id, occurrence)
        keys.append((scene, int(view), int(frame), track_id, occurrence))
        previous = history.get(token)
        # Exact MIA state split in get_matched_ids: a target is new when its
        # tracker ID exceeds the maximum ID committed before this frame.
        is_new[index] = previous_max_id is None or track_id > previous_max_id
        if previous is not None and previous[0] == frame - 1:
            trajectory[index, :4] = spatial[index] - previous[1]
            trajectory[index, 4] = 1.0
    for index, key in enumerate(keys):
        token = (key[0], key[1], key[3], key[4])
        history[token] = (frame, spatial[index].clone())
    return keys, boxes, spatial, trajectory, is_new


def positive_pairs(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    right_lookup = {int(label): index for index, label in enumerate(right.tolist()) if label >= 0}
    pairs = [(index, right_lookup[int(label)]) for index, label in enumerate(left.tolist())
             if int(label) >= 0 and int(label) in right_lookup]
    return torch.tensor(pairs, dtype=torch.long).reshape(-1, 2)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mia-results", required=True, type=Path)
    parser.add_argument("--geometry-trace", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--frontend", default="autoassign_bytetrack",
                        choices=("autoassign_bytetrack", "carafe_bytetrack"))
    parser.add_argument("--split", required=True, choices=("train", "val", "test"))
    parser.add_argument("--data", default="data/datafull")
    parser.add_argument("--xml", default="data/datafull/new_xml")
    parser.add_argument("--split-file", default="configs/splits.json")
    parser.add_argument("--scenes", default="", help="Optional comma-separated split subset")
    parser.add_argument("--truth-iou", type=float, default=0.5)
    parser.add_argument(
        "--insertion-point",
        choices=("post_mia_output", "pre_cross_view_id_allocation"),
        default="post_mia_output",
    )
    parser.add_argument(
        "--pooling",
        choices=("per_level_roi_align_1x1_then_equal_fpn_mean",
                 "assigned_fpn_roi_align_3x3_meanmax"),
        default="per_level_roi_align_1x1_then_equal_fpn_mean",
    )
    parser.add_argument("--without-truth", action="store_true",
                        help="Required for the final test feature build before frozen evaluation")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-frames", type=int, default=0,
                        help="Smoke-test limit per scene; zero processes the complete split")
    args = parser.parse_args()

    pairs = select_scene_pairs(
        discover_scene_pairs(Path(args.data) / args.split, args.xml), args.split,
        split_file=args.split_file,
    )
    selected_scenes = {value.strip() for value in args.scenes.split(",") if value.strip()}
    if selected_scenes:
        pairs = [pair for pair in pairs if str(pair.scene_id) in selected_scenes]
    model = build(args.frontend, args.device)
    from mmdet.datasets.pipelines import Compose
    pipeline = Compose(model.cfg.data.test.pipeline)
    samples = []
    processed = 0
    for pair in pairs:
        scene = str(pair.scene_id)
        payloads = [
            json.loads((args.mia_results / f"{scene}-1.json").read_text(encoding="utf-8")),
            json.loads((args.mia_results / f"{scene}-2.json").read_text(encoding="utf-8")),
        ]
        traces = json.loads((args.geometry_trace / f"{scene}.json").read_text(encoding="utf-8"))
        images = [list_frame_files(pair.view1_frames), list_frame_files(pair.view2_frames)]
        truths = ([parse_cvat_tracks(pair.view1_xml), parse_cvat_tracks(pair.view2_xml)]
                  if not args.without_truth else [None, None])
        history = {}
        previous_max_ids = [None, None]
        count = min(len(images[0]), len(images[1]))
        if args.max_frames:
            count = min(count, args.max_frames)
        for frame in range(count):
            rows = [payloads[view].get(f"frame={frame}", []) for view in range(2)]
            trace = traces[f"frame={frame}"]
            exact_previous = trace.get("previous_max_ids", {})
            frame_previous_max_ids = previous_max_ids
            if args.insertion_point == "pre_cross_view_id_allocation" and exact_previous:
                frame_previous_max_ids = [
                    float(exact_previous["view1"]), float(exact_previous["view2"])
                ]
            node_data = [
                frame_nodes(
                    rows[view], scene, view + 1, frame, history,
                    frame_previous_max_ids[view],
                )
                for view in range(2)
            ]
            for view in range(2):
                if rows[view]:
                    current_max = max(int(row[0]) for row in rows[view])
                    previous = previous_max_ids[view]
                    previous_max_ids[view] = current_max if previous is None else max(previous, current_max)
            appearance = []
            for view in range(2):
                boxes = node_data[view][1]
                if len(boxes):
                    feature = frame_features(
                        model, pipeline, images[view][frame], boxes.tolist(), pooling=args.pooling
                    ).half()
                else:
                    feature_dim = 512 if args.pooling == "assigned_fpn_roi_align_3x3_meanmax" else 256
                    feature = torch.empty((0, feature_dim), dtype=torch.float16)
                appearance.append(feature)

            labels = []
            for view in range(2):
                if args.without_truth:
                    labels.append(torch.full((len(rows[view]),), -1, dtype=torch.long))
                else:
                    labels.append(match_truth(node_data[view][1], truths[view].get(frame, []), args.truth_iou))
            pairs_at_t = (torch.empty((0, 2), dtype=torch.long) if args.without_truth
                          else positive_pairs(labels[0], labels[1]))

            left_boxes, right_boxes = node_data[0][1], node_data[1][1]
            if len(left_boxes) and len(right_boxes):
                h12 = np.asarray(trace["view1_to_view2"], dtype=np.float64)
                h21 = np.asarray(trace["view2_to_view1"], dtype=np.float64)
                centers_left = np.column_stack(((left_boxes[:, 0] + left_boxes[:, 2]) * 0.5,
                                                (left_boxes[:, 1] + left_boxes[:, 3]) * 0.5))
                centers_right = np.column_stack(((right_boxes[:, 0] + right_boxes[:, 2]) * 0.5,
                                                 (right_boxes[:, 1] + right_boxes[:, 3]) * 0.5))
                mapped_left = cv2.perspectiveTransform(
                    centers_left[:, None, :].astype(np.float32), h12
                )[:, 0]
                mapped_right = cv2.perspectiveTransform(
                    centers_right[:, None, :].astype(np.float32), h21
                )[:, 0]
                distance_ab = np.linalg.norm(
                    mapped_left[:, None, :] - centers_right[None, :, :], axis=2
                )
                distance_ba = np.linalg.norm(
                    mapped_right[:, None, :] - centers_left[None, :, :], axis=2
                )
                scales_left = np.where(node_data[0][4].numpy(), 80.0, 50.0)
                scales_right = np.where(node_data[1][4].numpy(), 80.0, 50.0)
                geometry = torch.as_tensor(
                    np.exp(-distance_ab / scales_left[:, None]), dtype=torch.float32
                )
                reverse_geometry = torch.as_tensor(
                    np.exp(-distance_ba / scales_right[:, None]), dtype=torch.float32
                )
                reachable_ab = mapped_box_reachable(left_boxes, h12)
                reachable_ba = mapped_box_reachable(right_boxes, h21)
                # Geometry is a soft feature, never a candidate-deletion rule.
                # Homography projection outside the image is still useful as
                # a diagnostic state, but hard-pruning the complete source row
                # removes many difficult true repairs before topology sees
                # them.  Keep every synchronized target pair trainable.
                candidate_mask = torch.ones_like(geometry, dtype=torch.bool)
                reverse_candidate_mask = torch.ones_like(
                    reverse_geometry, dtype=torch.bool
                )
            else:
                geometry = torch.empty((len(left_boxes), len(right_boxes)), dtype=torch.float32)
                reverse_geometry = torch.empty((len(right_boxes), len(left_boxes)), dtype=torch.float32)
                candidate_mask = torch.empty_like(geometry, dtype=torch.bool)
                reverse_candidate_mask = torch.empty_like(reverse_geometry, dtype=torch.bool)

            samples.append({
                "scene_id": scene, "frame": frame,
                "left_keys": node_data[0][0], "right_keys": node_data[1][0],
                "left": (appearance[0], node_data[0][2], node_data[0][3]),
                "right": (appearance[1], node_data[1][2], node_data[1][3]),
                "positive_pairs": pairs_at_t,
                "geometric_score": geometry,
                "reverse_geometric_score": reverse_geometry,
                "candidate_mask": candidate_mask,
                "reverse_candidate_mask": reverse_candidate_mask,
                "left_geometry_reachable": torch.as_tensor(
                    reachable_ab, dtype=torch.bool
                ) if len(left_boxes) and len(right_boxes) else torch.empty(
                    len(left_boxes), dtype=torch.bool
                ),
                "right_geometry_reachable": torch.as_tensor(
                    reachable_ba, dtype=torch.bool
                ) if len(left_boxes) and len(right_boxes) else torch.empty(
                    len(right_boxes), dtype=torch.bool
                ),
                "left_new_target": node_data[0][4],
                "right_new_target": node_data[1][4],
            })
            processed += 1
            if processed == 1 or processed % 100 == 0:
                print(f"synchronous official-FPN frames={processed}", flush=True)

    checkpoint = str(detector_checkpoint(args.frontend))
    payload = {
        "metadata": {
            "version": 1,
            "representation": "mia_net_synchronous_frame_targets",
            "insertion_point": args.insertion_point,
            "frontend": args.frontend,
            "appearance_source": "frozen_official_mia_detector_fpn",
            "detector_checkpoint": checkpoint,
            "pooling": args.pooling,
            "temporal_aggregation": "current_frame_only",
            "topology_time": "all nodes are targets observed in the same synchronized frame t",
            "trajectory": "normalized box state at t minus t-1 plus binary validity",
            "geometry_direction": "ordered bidirectional view1_to_view2 and view2_to_view1",
            "geometry_score": "exp(-directed_mapped_center_distance/state_scale)",
            "geometry_scales": {"new": 80.0, "existing": 50.0},
            "geometry_state": "official MIA new iff current ID exceeds previous committed maximum ID",
            "candidate_rule": "all synchronized target pairs; geometry is soft and never deletes candidates",
            "geometry_reachability": "diagnostic only; never used as a candidate mask",
            "truth_labels_present": not args.without_truth,
            "split": args.split,
        },
        "samples": samples,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.out)
    print(f"saved synchronous frames={len(samples)} to {args.out}")


if __name__ == "__main__":
    main()
