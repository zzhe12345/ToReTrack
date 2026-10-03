from __future__ import annotations

"""Render sequence-level ID-correctness colors against CVAT ground truth."""

from collections import Counter, defaultdict
import csv
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

from uav_tracking.dataset import list_frame_files, parse_cvat_tracks


GREEN = (0, 205, 0)       # correct association
RED = (0, 0, 255)         # wrong identity or false positive
YELLOW = (0, 220, 255)    # missed ground-truth target
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)


def box_iou(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left = np.asarray(left, dtype=np.float32).reshape(-1, 4)
    right = np.asarray(right, dtype=np.float32).reshape(-1, 4)
    if not len(left) or not len(right):
        return np.zeros((len(left), len(right)), dtype=np.float32)
    top_left = np.maximum(left[:, None, :2], right[None, :, :2])
    bottom_right = np.minimum(left[:, None, 2:], right[None, :, 2:])
    size = np.maximum(bottom_right - top_left, 0.0)
    intersection = size[..., 0] * size[..., 1]
    area_left = np.maximum(left[:, 2] - left[:, 0], 0.0) * np.maximum(
        left[:, 3] - left[:, 1], 0.0
    )
    area_right = np.maximum(right[:, 2] - right[:, 0], 0.0) * np.maximum(
        right[:, 3] - right[:, 1], 0.0
    )
    union = area_left[:, None] + area_right[None, :] - intersection
    return intersection / np.maximum(union, 1e-8)


def frame_match(predictions: list[list[float]], truths: list[dict], threshold: float):
    pred_boxes = np.asarray([row[1:5] for row in predictions], dtype=np.float32).reshape(-1, 4)
    truth_boxes = np.asarray([row["bbox"] for row in truths], dtype=np.float32).reshape(-1, 4)
    overlap = box_iou(pred_boxes, truth_boxes)
    matched = []
    if len(pred_boxes) and len(truth_boxes):
        rows, columns = linear_sum_assignment(1.0 - overlap)
        matched = [
            (int(row), int(column), float(overlap[row, column]))
            for row, column in zip(rows, columns)
            if overlap[row, column] >= threshold
        ]
    matched_pred = {row for row, _, _ in matched}
    matched_truth = {column for _, column, _ in matched}
    return matched, matched_pred, matched_truth


def global_identity_mapping(payloads, truths_by_view, threshold: float) -> dict[int, int]:
    counts: Counter[tuple[int, int]] = Counter()
    prediction_ids = set()
    truth_ids = set()
    for view in (0, 1):
        payload = payloads[view]
        for frame_key, predictions in payload.items():
            frame = int(frame_key.split("=", 1)[1])
            truths = truths_by_view[view].get(frame, [])
            matched, _, _ = frame_match(predictions, truths, threshold)
            for pred_index, truth_index, _ in matched:
                pred_id = int(predictions[pred_index][0])
                truth_id = int(truths[truth_index]["track_id"])
                prediction_ids.add(pred_id)
                truth_ids.add(truth_id)
                counts[(pred_id, truth_id)] += 1
    prediction_ids = sorted(prediction_ids)
    truth_ids = sorted(truth_ids)
    if not prediction_ids or not truth_ids:
        return {}
    matrix = np.zeros((len(prediction_ids), len(truth_ids)), dtype=np.float64)
    pred_index = {value: index for index, value in enumerate(prediction_ids)}
    truth_index = {value: index for index, value in enumerate(truth_ids)}
    for (pred_id, truth_id), count in counts.items():
        matrix[pred_index[pred_id], truth_index[truth_id]] = count
    rows, columns = linear_sum_assignment(-matrix)
    return {
        prediction_ids[row]: truth_ids[column]
        for row, column in zip(rows, columns)
        if matrix[row, column] > 0
    }


def clipped_box(box, width: int, height: int):
    x1, y1, x2, y2 = (int(round(float(value))) for value in box)
    return (
        max(0, min(width - 1, x1)),
        max(0, min(height - 1, y1)),
        max(0, min(width - 1, x2)),
        max(0, min(height - 1, y2)),
    )


def draw_label(image, box, label: str, color) -> None:
    height, width = image.shape[:2]
    x1, y1, x2, y2 = clipped_box(box, width, height)
    thickness = max(2, round(min(width, height) / 450))
    cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)
    # Dense UAV scenes can contain well over one hundred targets.  Keep the
    # label compact so IDs remain readable without hiding most of the frame.
    font_scale = max(0.42, min(width, height) / 2100.0)
    text_thickness = max(1, thickness - 1)
    (text_width, text_height), baseline = cv2.getTextSize(
        label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_thickness
    )
    label_y1 = max(0, y1 - text_height - baseline - 6)
    label_y2 = min(height - 1, label_y1 + text_height + baseline + 6)
    label_x2 = min(width - 1, x1 + text_width + 8)
    cv2.rectangle(image, (x1, label_y1), (label_x2, label_y2), color, -1)
    text_color = BLACK if color == YELLOW else WHITE
    cv2.putText(
        image, label, (x1 + 4, label_y2 - baseline - 3),
        cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color,
        text_thickness, cv2.LINE_AA,
    )


def draw_header(image, sequence: int, view: int, frame: int, counts: Counter) -> None:
    overlay = image.copy()
    cv2.rectangle(overlay, (0, 0), (image.shape[1], 82), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.75, image, 0.25, 0.0, image)
    title = f"Sequence {sequence} | View {view} | Frame {frame}"
    legend = (
        f"GREEN correct: {counts['correct']}   RED wrong/FP: "
        f"{counts['wrong_identity'] + counts['false_positive']}   "
        f"YELLOW missed: {counts['missed']}"
    )
    cv2.putText(image, title, (18, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.78, WHITE, 2, cv2.LINE_AA)
    cv2.putText(image, legend, (18, 65), cv2.FONT_HERSHEY_SIMPLEX,
                0.62, WHITE, 2, cv2.LINE_AA)


def open_video(path: Path, width: int, height: int, fps: float):
    for codec, suffix in (("mp4v", ".mp4"), ("MJPG", ".avi")):
        candidate = path.with_suffix(suffix)
        writer = cv2.VideoWriter(
            str(candidate), cv2.VideoWriter_fourcc(*codec), fps, (width, height)
        )
        if writer.isOpened():
            return writer, candidate
        writer.release()
    raise RuntimeError("无法创建视频文件（mp4v/MJPG 编码均不可用）")


def render_sequence(*, sequence: int, split: str, result_dir: Path,
                    output_dir: Path, iou_threshold: float = 0.5,
                    fps: float = 20.0, save_frames: bool = True,
                    max_frames: int = 0, data_root: Path | None = None) -> dict:
    if data_root is None:
        raise ValueError("data_root is required")
    data_root = Path(data_root).resolve()
    frame_dirs = [
        data_root / split / str(view) / f"{sequence}-{view}"
        for view in (1, 2)
    ]
    xml_paths = [
        data_root / "new_xml" / str(view) / f"{sequence}-{view}.xml"
        for view in (1, 2)
    ]
    frames_by_view = [list_frame_files(path) for path in frame_dirs]
    if max_frames:
        frames_by_view = [frames[:max_frames] for frames in frames_by_view]
    payloads = [
        json.loads((result_dir / f"{sequence}-{view}.json").read_text(encoding="utf-8"))
        for view in (1, 2)
    ]
    truths_by_view = [parse_cvat_tracks(path) for path in xml_paths]
    identity_map = global_identity_mapping(payloads, truths_by_view, iou_threshold)

    output_dir.mkdir(parents=True, exist_ok=True)
    summary = Counter()
    view_summaries = {}
    detail_rows = []
    video_paths = []
    for view in (0, 1):
        view_number = view + 1
        frame_paths = frames_by_view[view]
        if not frame_paths:
            raise FileNotFoundError(f"视角 {view_number} 没有输入帧")
        first = cv2.imread(str(frame_paths[0]))
        if first is None:
            raise OSError(f"无法读取图像：{frame_paths[0]}")
        height, width = first.shape[:2]
        writer, video_path = open_video(
            output_dir / f"sequence_{sequence}_view{view_number}", width, height, fps
        )
        video_paths.append(str(video_path.resolve()))
        frame_output = output_dir / f"view{view_number}_frames"
        if save_frames:
            frame_output.mkdir(parents=True, exist_ok=True)
        view_counts = Counter()
        payload = payloads[view]
        truths = truths_by_view[view]
        try:
            for frame, image_path in enumerate(frame_paths):
                image = cv2.imread(str(image_path))
                if image is None:
                    raise OSError(f"无法读取图像：{image_path}")
                predictions = payload.get(f"frame={frame}", [])
                truth_rows = truths.get(frame, [])
                matched, matched_pred, matched_truth = frame_match(
                    predictions, truth_rows, iou_threshold
                )
                frame_counts = Counter()
                for pred_index, truth_index, overlap in matched:
                    prediction = predictions[pred_index]
                    truth = truth_rows[truth_index]
                    pred_id = int(prediction[0])
                    truth_id = int(truth["track_id"])
                    correct = identity_map.get(pred_id) == truth_id
                    status = "correct" if correct else "wrong_identity"
                    color = GREEN if correct else RED
                    label = (
                        f"ID:{pred_id}"
                        if correct else f"ERR:{pred_id}/GT:{truth_id}"
                    )
                    draw_label(image, prediction[1:5], label, color)
                    frame_counts[status] += 1
                    detail_rows.append({
                        "sequence": sequence, "view": view_number, "frame": frame,
                        "status": status, "predicted_id": pred_id,
                        "mapped_gt_id": identity_map.get(pred_id, ""),
                        "matched_gt_id": truth_id, "iou": f"{overlap:.6f}",
                        "x1": prediction[1], "y1": prediction[2],
                        "x2": prediction[3], "y2": prediction[4],
                    })
                for pred_index, prediction in enumerate(predictions):
                    if pred_index in matched_pred:
                        continue
                    pred_id = int(prediction[0])
                    draw_label(image, prediction[1:5], f"FP:{pred_id}", RED)
                    frame_counts["false_positive"] += 1
                    detail_rows.append({
                        "sequence": sequence, "view": view_number, "frame": frame,
                        "status": "false_positive", "predicted_id": pred_id,
                        "mapped_gt_id": identity_map.get(pred_id, ""),
                        "matched_gt_id": "", "iou": "",
                        "x1": prediction[1], "y1": prediction[2],
                        "x2": prediction[3], "y2": prediction[4],
                    })
                for truth_index, truth in enumerate(truth_rows):
                    if truth_index in matched_truth:
                        continue
                    truth_id = int(truth["track_id"])
                    draw_label(image, truth["bbox"], f"MISS:{truth_id}", YELLOW)
                    frame_counts["missed"] += 1
                    bbox = truth["bbox"]
                    detail_rows.append({
                        "sequence": sequence, "view": view_number, "frame": frame,
                        "status": "missed", "predicted_id": "",
                        "mapped_gt_id": "", "matched_gt_id": truth_id, "iou": "",
                        "x1": bbox[0], "y1": bbox[1], "x2": bbox[2], "y2": bbox[3],
                    })
                view_counts.update(frame_counts)
                draw_header(image, sequence, view_number, frame, frame_counts)
                writer.write(image)
                if save_frames:
                    cv2.imwrite(str(frame_output / f"{frame:06d}.jpg"), image,
                                [cv2.IMWRITE_JPEG_QUALITY, 92])
                if frame == 0 or (frame + 1) % 100 == 0 or frame + 1 == len(frame_paths):
                    print(
                        f"可视化 view={view_number} frame={frame + 1}/{len(frame_paths)}",
                        flush=True,
                    )
        finally:
            writer.release()
        view_summaries[str(view_number)] = dict(view_counts)
        summary.update(view_counts)

    columns = [
        "sequence", "view", "frame", "status", "predicted_id",
        "mapped_gt_id", "matched_gt_id", "iou", "x1", "y1", "x2", "y2",
    ]
    with (output_dir / "match_details.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(detail_rows)
    mapping_json = {str(key): int(value) for key, value in sorted(identity_map.items())}
    final = {
        "sequence": sequence,
        "split": split,
        "iou_threshold": iou_threshold,
        "identity_mapping_predicted_to_ground_truth": mapping_json,
        "correct": int(summary["correct"]),
        "wrong_identity": int(summary["wrong_identity"]),
        "false_positive": int(summary["false_positive"]),
        "wrong_total": int(summary["wrong_identity"] + summary["false_positive"]),
        "missed": int(summary["missed"]),
        "per_view": view_summaries,
        "videos": video_paths,
        "color_definition": {
            "green": "IoU matched and sequence-level global ID mapping is correct",
            "red": "wrong global identity or unmatched false-positive prediction",
            "yellow": "ground-truth target with no IoU-matched prediction",
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(final, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return final
