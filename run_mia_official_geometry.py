from __future__ import annotations

"""Run the downloaded MIA-Net cross-view ID-allocation path on MDMT.

This is a non-visual, path-portable transcription of the inference loop in
``demo/multiDrone_matchingIDallocation-NMS.py``.  It deliberately keeps the
official first-frame GT initialization, bilateral tracker feedback, geometric
ID refresh, and 0.3-IoU track NMS.
"""

import argparse
import copy
import contextlib
import faulthandler
import io
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from torchvision.ops import box_iou

from run_mia_frontend import build
from uav_tracking.mmcv_ops_compat import install as install_mmcv_ops


class _DiscardOutput:
    """File-like sink that suppresses verbose official prints without buffering."""

    def write(self, value):
        return len(value)

    def flush(self):
        return None


ROOT = Path(__file__).resolve().parent
DEMO = ROOT / "third_party" / "mia_net_official" / "demo"
sys.path.insert(0, str(DEMO))
sys.path.insert(0, str(DEMO / "utils"))
install_mmcv_ops()

from utils.common import (  # noqa: E402
    A_same_target_refresh_same_ID,
    B_same_target_refresh_same_ID,
    all_nms,
    get_matched_ids,
    get_matched_ids_frame1,
    read_xml_r,
    same_target_refresh_same_ID,
)
from utils.matching_pure import calculate_cent_corner_pst, matching  # noqa: E402
from utils.supplement import not_matched_supplement  # noqa: E402




class CachedDetectionFusion:
    """Fuse a frozen secondary detector cache into official AutoAssign."""

    def __init__(self, root: Path, threshold: float = 0.65,
                 official_iou: float = 0.3, maximum: int = 100,
                 track_only: bool = False, injected_score: float = 0.59,
                 replace: bool = False):
        self.root = root
        self.threshold = float(threshold)
        self.official_iou = float(official_iou)
        self.maximum = int(maximum)
        self.track_only = bool(track_only)
        self.injected_score = float(injected_score)
        self.replace = bool(replace)
        self.rows = {}
        self.stats = {"frames": 0, "candidates": 0, "above_threshold": 0,
                      "nonoverlap": 0, "added": 0, "matched_tracks": 0}

    def start_scene(self, scene: str):
        self.rows = {
            view: torch.load(
                self.root / f"scene{scene}_view{view}.pt",
                map_location="cpu", weights_only=False,
            )
            for view in (1, 2)
        }

    @torch.inference_mode()
    def add(self, official: torch.Tensor, view: int, frame: int) -> torch.Tensor:
        row = self.rows[view][frame]
        boxes = torch.as_tensor(row["boxes"], device=official.device,
                                dtype=official.dtype).reshape(-1, 4)
        scores = torch.as_tensor(row["scores"], device=official.device,
                                 dtype=official.dtype).flatten()
        self.stats["frames"] += 1
        self.stats["candidates"] += int(len(boxes))
        selected = torch.nonzero(scores >= self.threshold, as_tuple=False).flatten()
        self.stats["above_threshold"] += int(len(selected))
        if len(selected) and len(official) and not self.replace:
            selected = selected[
                (box_iou(boxes[selected], official[:, :4]).amax(dim=1)
                 if len(official) else boxes.new_zeros(len(selected)))
                < self.official_iou
            ]
        self.stats["nonoverlap"] += int(len(selected))
        if self.maximum and len(selected) > self.maximum:
            selected = selected[torch.topk(scores[selected], self.maximum).indices]
        addition_scores = (
            scores.new_full((len(selected),), self.injected_score)
            if self.track_only else scores[selected]
        )
        addition = torch.cat([boxes[selected], addition_scores[:, None]], dim=1)
        self.stats["added"] += int(len(addition))
        return addition if self.replace else torch.cat([official, addition], dim=0)


def inference_bilateral(model, image_path: Path, frame_id: int, bboxes, ids, labels,
                        max_id: int,
                        external: CachedDetectionFusion | None = None,
                        external_view: int = 0):
    from mmcv.parallel import collate, scatter
    from mmdet.datasets.pipelines import Compose

    data = dict(img_info=dict(filename=str(image_path), frame_id=frame_id), img_prefix=None)
    data = Compose(model.cfg.data.test.pipeline)(data)
    data = collate([data], samples_per_gpu=1)
    device = next(model.parameters()).device
    data = scatter(data, [device])[0]
    bil = dict(bboxes1=bboxes, ids1=ids, labels1=labels, max_id=max_id)
    with torch.inference_mode():
        if external is not None:
            image = data["img"][0] if isinstance(data["img"], (list, tuple)) else data["img"]
            metas = (data["img_metas"][0]
                     if isinstance(data["img_metas"], (list, tuple)) else data["img_metas"])
            while isinstance(metas, (list, tuple)):
                metas = metas[0]
            pyramid = model.detector.extract_feat(image)
            detector_outputs = model.detector.bbox_head(pyramid)
            det_bboxes, det_labels = model.detector.bbox_head.get_bboxes(
                *detector_outputs, img_metas=[metas],
                cfg=model.detector.test_cfg, rescale=True, with_nms=True,
            )[0]
            from mmdet.core import bbox2result
            detector_results = bbox2result(
                det_bboxes, det_labels, model.detector.bbox_head.num_classes
            )
            official_np = np.concatenate(detector_results, axis=0)
            official = torch.as_tensor(official_np, device=image.device, dtype=image.dtype)
            detections = official
            if external is not None:
                detections = external.add(detections, external_view, frame_id)
            det_labels = torch.zeros(len(detections), dtype=torch.long, device=image.device)
            track_bboxes, track_labels, track_ids, next_max = model.tracker.track(
                img=image, img_metas=[metas], model=model, bboxes=detections,
                labels=det_labels, frame_id=frame_id, rescale=True, bil=bil,
            )
            if external is not None and external.track_only and len(track_bboxes):
                external.stats["matched_tracks"] += int(torch.isclose(
                    track_bboxes[:, -1],
                    track_bboxes.new_tensor(external.injected_score), atol=1e-5,
                ).sum().item())
            from mmtrack.core import outs2results
            track_results = outs2results(
                bboxes=track_bboxes, labels=track_labels, ids=track_ids, num_classes=1
            )
            visible_detections = (
                detections if external is not None and not external.track_only
                else official
            )
            visible_labels = torch.zeros(
                len(visible_detections), dtype=torch.long, device=image.device
            )
            det_results = outs2results(
                bboxes=visible_detections, labels=visible_labels, num_classes=1
            )
            return {"det_bboxes": det_results["bbox_results"],
                    "track_bboxes": track_results["bbox_results"]}, next_max
        return model(return_loss=False, rescale=True, frame_id=frame_id, bil=bil, **data)


def centers_and_corners(image, rows):
    if len(rows) == 0:
        return np.empty((0, 2), dtype=np.float32), np.empty((0, 2), dtype=np.float32)
    return calculate_cent_corner_pst(image, rows)


def transformation(src, dst, previous, image1, image2):
    """Official ``supp_compute_transf_matrix`` behavior without debug output."""
    if len(src) >= 5:
        current, _ = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
        if current is None:
            current = previous.copy()
        return current, current.copy()

    # This is the official global-feature fallback (three estimates followed
    # by its cosine-consistency selection).
    matrices = [matching(image1, image2) for _ in range(3)]
    first, second, third = matrices
    if isinstance(first, int):
        return previous.copy(), previous.copy()

    def cosine(a, b):
        denom = np.linalg.norm(a.reshape(1, -1)) * np.linalg.norm(b.reshape(1, -1))
        return float(a.reshape(1, -1).dot(b.reshape(1, -1).T)[0, 0] / max(denom, 1e-12))

    if cosine(first, previous) > 0.99:
        chosen = first
    elif not isinstance(second, int) and cosine(first, second) > 0.99:
        chosen = (first + second) / 2
    elif not isinstance(third, int) and cosine(first, third) > 0.99:
        chosen = (first + third) / 2
    elif not isinstance(second, int) and not isinstance(third, int) and cosine(second, third) > 0.99:
        chosen = (second + third) / 2
    else:
        chosen = previous
    return chosen, chosen.copy()


def run_scene(model1, model2, data: Path, xml_dir: Path, scene: str, max_frames: int,
              capture_detections: bool = False,
              external: CachedDetectionFusion | None = None,
              timing_samples: list[float] | None = None,
              timing_checkpoint: Path | None = None):
    frames1 = sorted((data / "1" / f"{scene}-1").glob("*.jpg"), key=lambda p: int(p.stem))
    frames2 = sorted((data / "2" / f"{scene}-2").glob("*.jpg"), key=lambda p: int(p.stem))
    if max_frames:
        frames1, frames2 = frames1[:max_frames], frames2[:max_frames]
    if len(frames1) != len(frames2):
        raise RuntimeError(f"scene {scene}: view lengths differ ({len(frames1)} vs {len(frames2)})")

    model1.tracker.reset()
    model2.tracker.reset()
    if external is not None:
        external.start_scene(scene)
    payload1, payload2, geometry_trace = {}, {}, {}
    pre_match_payload1, pre_match_payload2 = {}, {}
    detection_trace1, detection_trace2 = [], []
    matched_ids = []
    a_max = b_max = 0
    bboxes1, ids1, labels1 = read_xml_r(str(xml_dir / "1" / f"{scene}-1.xml"), 0)
    bboxes2, ids2, labels2 = read_xml_r(str(xml_dir / "2" / f"{scene}-2.xml"), 0)
    f1_last = f2_last = None

    for frame_id, (path1, path2) in enumerate(zip(frames1, frames2)):
        if timing_samples is not None and torch.cuda.is_available():
            torch.cuda.synchronize()
        frame_started = time.perf_counter() if timing_samples is not None else None
        image1, image2 = cv2.imread(str(path1)), cv2.imread(str(path2))
        max_id = int(max(a_max, b_max))
        result1, max_id = inference_bilateral(
            model1, path1, frame_id, bboxes1, ids1, labels1, max_id,
            external, 1,
        )
        result2, max_id = inference_bilateral(
            model2, path2, frame_id, bboxes2, ids2, labels2, max_id,
            external, 2,
        )
        det1 = result1["det_bboxes"][0]
        det2 = result2["det_bboxes"][0]
        if capture_detections:
            # Preserve the exact AutoAssign post-NMS detector proposals seen by
            # the official supplementation code.  This trace is diagnostic only
            # and does not alter ByteTrack or cross-view state.
            detection_trace1.append(np.asarray(det1, dtype=np.float32).copy())
            detection_trace2.append(np.asarray(det2, dtype=np.float32).copy())
        tracks1 = np.asarray(result1["track_bboxes"][0])
        tracks2 = np.asarray(result2["track_bboxes"][0])
        # Exact insertion-point state: detector + single-view ByteTrack have
        # completed, but no cross-view MIA ID refresh, completion, or NMS has
        # yet modified the current frame.
        pre_match_payload1[f"frame={frame_id}"] = tracks1[:, :5].copy().tolist()
        pre_match_payload2[f"frame={frame_id}"] = tracks2[:, :5].copy().tolist()
        previous_a_max, previous_b_max = float(a_max), float(b_max)
        cent1, corner1 = centers_and_corners(image1, tracks1)
        cent2, corner2 = centers_and_corners(image2, tracks2)

        if frame_id == 0:
            pts1, pts2, matched_ids = get_matched_ids_frame1(tracks1, tracks2, cent1, cent2)
            a_max = float(np.max(tracks1[:, 0]))
            b_max = float(np.max(tracks2[:, 0]))
            if len(pts1) < 5:
                raise RuntimeError(f"scene {scene}: fewer than five shared GT IDs in frame 0")
            f1_last, _ = cv2.findHomography(pts1, pts2, cv2.RANSAC, 5.0)
            f2_last, _ = cv2.findHomography(pts2, pts1, cv2.RANSAC, 5.0)
            if f1_last is None or f2_last is None:
                raise RuntimeError(f"scene {scene}: first-frame homography failed")
            f1, f2 = f1_last, f2_last
        else:
            confirmed = []
            values = get_matched_ids(
                tracks1, tracks2, cent1, cent2, corner1, corner2, a_max, b_max, confirmed
            )
            (matched_ids, pts1, pts2, a_new, a_pts, a_corners, b_new, b_pts,
             b_corners, a_old, a_old_pts, a_old_corners, _b_old, _b_old_pts,
             _b_old_corners) = values
            # ``supp_compute_transf_matrix`` crashes when SIFT returns its
            # integer failure sentinel.  ``transformation`` is the same three-
            # estimate/cosine selection with the intended last-matrix fallback.
            f1, f1_last = transformation(pts1, pts2, f1_last, image1, image2)
            tracks1, tracks2, matched_ids, _, confirmed = A_same_target_refresh_same_ID(
                a_new, a_pts, a_corners, f1, cent2, tracks1, tracks2,
                matched_ids, det1, det2, image2, confirmed, thres=80
            )

            values = get_matched_ids(
                tracks1, tracks2, cent1, cent2, corner1, corner2, a_max, b_max, confirmed
            )
            (matched_ids, pts1, pts2, _a_new, _a_pts, _a_corners, b_new, b_pts,
             b_corners, _a_old, _a_old_pts, _a_old_corners, _b_old,
             _b_old_pts, _b_old_corners) = values
            f2, f2_last = transformation(pts2, pts1, f2_last, image2, image1)
            tracks1, tracks2, matched_ids, _, confirmed = B_same_target_refresh_same_ID(
                b_new, b_pts, b_corners, f2, cent1, tracks1, tracks2,
                matched_ids, det1, det2, image1, confirmed, thres=80
            )

            values = get_matched_ids(
                tracks1, tracks2, cent1, cent2, corner1, corner2, a_max, b_max, confirmed
            )
            (matched_ids, _pts1, _pts2, _a_new, _a_pts, _a_corners, _b_new,
             _b_pts, _b_corners, a_old, a_old_pts, a_old_corners, b_old,
             b_old_pts, b_old_corners) = values
            tracks1, tracks2, matched_ids, _, confirmed = same_target_refresh_same_ID(
                a_old, a_old_pts, a_old_corners, f1, cent2, tracks1, tracks2,
                matched_ids, det1, det2, image2, confirmed, thres=50
            )

            # The paper's ``supplement_MIA.py`` performs two additional
            # cross-view completion passes before NMS.  These were absent from
            # the NMS-only entry point and materially affect MOTA/MDA.
            values = get_matched_ids(
                tracks1, tracks2, cent1, cent2, corner1, corner2,
                a_max, b_max, confirmed
            )
            (matched_ids, _pts1, _pts2, _a_new, _a_pts, _a_corners, _b_new,
             _b_pts, _b_corners, a_old, a_old_pts, a_old_corners, b_old,
             b_old_pts, b_old_corners) = values
            supplement2 = np.array([])
            tracks1, tracks2, matched_ids, _, confirmed, supplement2 = not_matched_supplement(
                a_old, a_old_pts, a_old_corners, f1, cent2, tracks1, tracks2,
                matched_ids, det1, det2, image2, confirmed, supplement2, thres=50
            )
            supplement1 = np.array([])
            tracks2, tracks1, matched_ids, _, confirmed, supplement1 = not_matched_supplement(
                b_old, b_old_pts, b_old_corners, f2, cent2, tracks2, tracks1,
                matched_ids, det2, det1, image1, confirmed, supplement1, thres=50
            )
            if len(tracks1):
                a_max = max(a_max, float(np.max(tracks1[:, 0])))
            if len(tracks2):
                b_max = max(b_max, float(np.max(tracks2[:, 0])))
            if len(tracks1):
                tracks1 = all_nms(tracks1, 0.3)
            if len(tracks2):
                tracks2 = all_nms(tracks2, 0.3)

        payload1[f"frame={frame_id}"] = tracks1[:, :5].tolist()
        payload2[f"frame={frame_id}"] = tracks2[:, :5].tolist()
        # Persist the exact bidirectional transforms used by this official MIA
        # frame.  The topology branch consumes these frozen values as S_geo;
        # it must not estimate a different homography implementation later.
        geometry_trace[f"frame={frame_id}"] = {
            "view1_to_view2": np.asarray(f1, dtype=np.float64).tolist(),
            "view2_to_view1": np.asarray(f2, dtype=np.float64).tolist(),
            "previous_max_ids": {"view1": previous_a_max, "view2": previous_b_max},
        }
        bboxes1 = torch.as_tensor(tracks1[:, 1:5], dtype=torch.long)
        ids1 = torch.as_tensor(tracks1[:, 0], dtype=torch.long)
        labels1 = torch.zeros_like(ids1)
        bboxes2 = torch.as_tensor(tracks2[:, 1:5], dtype=torch.long)
        ids2 = torch.as_tensor(tracks2[:, 0], dtype=torch.long)
        labels2 = torch.zeros_like(ids2)
        if timing_samples is not None:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            timing_samples.append(time.perf_counter() - frame_started)
            if timing_checkpoint is not None and (
                    (frame_id + 1) % 25 == 0 or frame_id + 1 == len(frames1)):
                timing_checkpoint.parent.mkdir(parents=True, exist_ok=True)
                timing_checkpoint.write_text(json.dumps({
                    "scene": scene,
                    "completed_paired_frames": frame_id + 1,
                    "scene_paired_frames": len(frames1),
                    "latest_frame_seconds": timing_samples[-1],
                    "cumulative_timed_seconds": float(sum(timing_samples)),
                    "updated_unix_seconds": time.time(),
                }, indent=2), encoding="utf-8")
        if frame_id % 50 == 0:
            print(f"official-mia scene={scene} frame={frame_id}/{len(frames1)}", flush=True)
    return (
        payload1, payload2, geometry_trace,
        pre_match_payload1, pre_match_payload2,
        detection_trace1, detection_trace2,
    )


def main():
    faulthandler.enable()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frontend", default="autoassign_bytetrack",
                        choices=("autoassign_bytetrack", "carafe_bytetrack"))
    parser.add_argument("--data", default="data/datafull/test")
    parser.add_argument("--xml-dir", default="data/datafull/new_xml")
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--geometry-trace-dir", default="",
        help="Optional directory for the exact per-frame MIA homographies used by the topology branch.",
    )
    parser.add_argument(
        "--pre-match-out", default="",
        help="Optional output for tracks immediately before cross-view ID allocation.",
    )
    parser.add_argument(
        "--detection-trace-dir", default="",
        help=(
            "Optional directory for the exact AutoAssign detector proposals "
            "consumed by the official paired-frame supplementation path."
        ),
    )
    parser.add_argument("--scenes", default="")
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--track-high", type=float, default=None)
    parser.add_argument("--track-low", type=float, default=None)
    parser.add_argument("--track-init", type=float, default=None)
    parser.add_argument("--track-match-high", type=float, default=None)
    parser.add_argument("--track-match-low", type=float, default=None)
    parser.add_argument("--track-match-tentative", type=float, default=None)
    parser.add_argument("--track-retain", type=int, default=None)
    parser.add_argument(
        "--test-scale", default="",
        help="Optional WIDTH,HEIGHT test scale; empty keeps official 1333,800.",
    )
    parser.add_argument("--external-detection-cache", default="")
    parser.add_argument("--external-detection-threshold", type=float, default=0.65)
    parser.add_argument("--external-official-iou", type=float, default=0.3)
    parser.add_argument("--external-maximum", type=int, default=100)
    parser.add_argument("--external-track-only", action="store_true")
    parser.add_argument("--external-injected-score", type=float, default=0.59)
    parser.add_argument("--external-replace", action="store_true")
    parser.add_argument(
        "--timing-report", default="",
        help="Optional JSON report with synchronized paired-frame latency and resource use.",
    )
    parser.add_argument(
        "--per-scene-timing-dir", default="",
        help="Optional directory that checkpoints a complete timing JSON after each scene.",
    )
    args = parser.parse_args()

    data, xml_dir, out = Path(args.data), Path(args.xml_dir), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    trace_dir = Path(args.geometry_trace_dir) if args.geometry_trace_dir else None
    if trace_dir is not None:
        trace_dir.mkdir(parents=True, exist_ok=True)
    pre_match_out = Path(args.pre_match_out) if args.pre_match_out else None
    if pre_match_out is not None:
        pre_match_out.mkdir(parents=True, exist_ok=True)
    detection_trace_dir = (
        Path(args.detection_trace_dir) if args.detection_trace_dir else None
    )
    if detection_trace_dir is not None:
        detection_trace_dir.mkdir(parents=True, exist_ok=True)
    selected = {item.strip() for item in args.scenes.split(",") if item.strip()}
    scenes = sorted((p.name[:-2] for p in (data / "1").glob("*-1") if p.is_dir()), key=int)
    if selected:
        scenes = [scene for scene in scenes if scene in selected]
    tracker_overrides = {
        "high": args.track_high, "low": args.track_low, "init": args.track_init,
        "match_high": args.track_match_high, "match_low": args.track_match_low,
        "match_tentative": args.track_match_tentative, "retain": args.track_retain,
    }
    test_scale = None
    if args.test_scale:
        values = tuple(int(value.strip()) for value in args.test_scale.split(","))
        if len(values) != 2:
            parser.error("--test-scale must be WIDTH,HEIGHT")
        test_scale = values
    build_started = time.perf_counter()
    model1 = build(args.frontend, args.device, tracker_overrides, test_scale)
    model2 = build(args.frontend, args.device, tracker_overrides, test_scale)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    build_seconds = time.perf_counter() - build_started
    single_model_parameters = sum(parameter.numel() for parameter in model1.parameters())
    external = None
    if args.external_detection_cache:
        if args.frontend != "autoassign_bytetrack":
            parser.error("external detection fusion is defined only for AutoAssign")
        external = CachedDetectionFusion(
            Path(args.external_detection_cache), args.external_detection_threshold,
            args.external_official_iou, args.external_maximum,
            args.external_track_only, args.external_injected_score,
            args.external_replace,
        )
    timing_samples: list[float] | None = [] if args.timing_report else None
    timing_checkpoint = (
        Path(args.timing_report).with_suffix(".progress.json")
        if args.timing_report else None
    )
    scene_timing: dict[str, dict[str, float | int]] = {}
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    processing_started = time.perf_counter()
    for scene in scenes:
        # The downloaded helpers are extremely verbose.  Suppressing only their
        # debug prints keeps the numerical path unchanged and the run auditable.
        sink = _DiscardOutput()
        scene_sample_start = len(timing_samples) if timing_samples is not None else 0
        with contextlib.redirect_stdout(sink):
            (payload1, payload2, geometry_trace, pre_match1, pre_match2,
             detection1, detection2) = run_scene(
                model1, model2, data, xml_dir, scene, args.max_frames,
                capture_detections=detection_trace_dir is not None,
                external=external,
                timing_samples=timing_samples,
                timing_checkpoint=timing_checkpoint,
            )
        if timing_samples is not None:
            values = timing_samples[scene_sample_start:]
            scene_timing[scene] = {
                "paired_frames": len(values),
                "processing_seconds": float(sum(values)),
                "paired_frame_latency_mean_ms": float(np.mean(values) * 1000.0),
                "paired_frame_latency_p50_ms": float(np.percentile(values, 50) * 1000.0),
                "paired_frame_latency_p95_ms": float(np.percentile(values, 95) * 1000.0),
            }
            if args.per_scene_timing_dir:
                scene_report = {
                    "scope": "two-view AutoAssign + ByteTrack + official MIA geometry",
                    "device": args.device,
                    "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
                    "scenes": [scene],
                    **scene_timing[scene],
                    "image_frames": len(values) * 2,
                    "paired_frames_per_second": float(len(values) / sum(values)),
                    "image_frames_per_second": float(len(values) * 2 / sum(values)),
                    "single_model_parameters": int(single_model_parameters),
                    "two_view_resident_parameters": int(single_model_parameters * 2),
                    "peak_cuda_allocated_bytes": (
                        int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else None
                    ),
                    "peak_cuda_reserved_bytes": (
                        int(torch.cuda.max_memory_reserved()) if torch.cuda.is_available() else None
                    ),
                }
                scene_path = Path(args.per_scene_timing_dir) / f"p0_scene{scene}_timing.json"
                scene_path.parent.mkdir(parents=True, exist_ok=True)
                scene_path.write_text(json.dumps(scene_report, indent=2), encoding="utf-8")
        print(f"official-mia scene={scene} complete", flush=True)
        (out / f"{scene}-1.json").write_text(json.dumps(payload1), encoding="utf-8")
        (out / f"{scene}-2.json").write_text(json.dumps(payload2), encoding="utf-8")
        if trace_dir is not None:
            (trace_dir / f"{scene}.json").write_text(
                json.dumps(geometry_trace), encoding="utf-8"
            )
        if pre_match_out is not None:
            (pre_match_out / f"{scene}-1.json").write_text(
                json.dumps(pre_match1), encoding="utf-8"
            )
            (pre_match_out / f"{scene}-2.json").write_text(
                json.dumps(pre_match2), encoding="utf-8"
            )
        if detection_trace_dir is not None:
            torch.save(
                detection1,
                detection_trace_dir / f"scene{scene}_view1.pt",
            )
            torch.save(
                detection2,
                detection_trace_dir / f"scene{scene}_view2.pt",
            )
        if external is not None:
            (out / "external_detection_stats.json").write_text(
                json.dumps(external.stats, indent=2), encoding="utf-8"
            )
    if args.timing_report:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        processing_wall_seconds = time.perf_counter() - processing_started
        values = np.asarray(timing_samples, dtype=np.float64)
        timed_seconds = float(values.sum())
        report = {
            "scope": "two-view AutoAssign + ByteTrack + official MIA geometry",
            "device": args.device,
            "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
            "scenes": scenes,
            "paired_frames": int(values.size),
            "image_frames": int(values.size * 2),
            "model_build_seconds": build_seconds,
            "processing_timed_seconds": timed_seconds,
            "processing_wall_seconds": processing_wall_seconds,
            "paired_frame_latency_mean_ms": float(values.mean() * 1000.0),
            "paired_frame_latency_p50_ms": float(np.percentile(values, 50) * 1000.0),
            "paired_frame_latency_p95_ms": float(np.percentile(values, 95) * 1000.0),
            "paired_frames_per_second": float(values.size / timed_seconds),
            "image_frames_per_second": float(values.size * 2 / timed_seconds),
            "single_model_parameters": int(single_model_parameters),
            "two_view_resident_parameters": int(single_model_parameters * 2),
            "peak_cuda_allocated_bytes": (
                int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else None
            ),
            "peak_cuda_reserved_bytes": (
                int(torch.cuda.max_memory_reserved()) if torch.cuda.is_available() else None
            ),
            "per_scene": scene_timing,
        }
        timing_path = Path(args.timing_report)
        timing_path.parent.mkdir(parents=True, exist_ok=True)
        timing_path.write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
