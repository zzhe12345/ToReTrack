from __future__ import annotations

"""Causally extend recently observed MIA tracks without future observations.

At frame t every proposal uses only detections at t and true observations from
frames <= t-1.  No proposal is inserted into the motion history, no future
endpoint is consulted, and previously emitted frames are never rewritten.
"""

import argparse
from collections import defaultdict, deque
import json
import time
from pathlib import Path

import cv2
import numpy as np

from uav_tracking.geometry import iou_matrix


def predict_box(history: deque, frame: int, velocity_decay: float,
                velocity_estimator: str = "last") -> np.ndarray | None:
    if not history:
        return None
    last_frame, last_box = history[-1]
    age = int(frame) - int(last_frame)
    if age <= 0:
        return None
    velocity = np.zeros(4, dtype=np.float64)
    if len(history) >= 2:
        if velocity_estimator == "last":
            previous_frame, previous_box = history[-2]
            delta = max(int(last_frame) - int(previous_frame), 1)
            velocity = (last_box - previous_box) / float(delta)
        elif velocity_estimator == "median":
            velocities = []
            for (first_frame, first_box), (second_frame, second_box) in zip(
                    history, list(history)[1:]):
                delta = max(int(second_frame) - int(first_frame), 1)
                velocities.append((second_box - first_box) / float(delta))
            velocity = np.median(np.stack(velocities), axis=0)
        elif velocity_estimator == "regression":
            times = np.asarray([int(value[0]) for value in history], dtype=np.float64)
            values = np.stack([value[1] for value in history])
            centered = times - times.mean()
            denominator = float(np.square(centered).sum())
            if denominator > 0.0:
                velocity = (centered[:, None] * values).sum(axis=0) / denominator
        else:
            raise ValueError(f"unsupported velocity estimator: {velocity_estimator}")
    if velocity_decay == 1.0:
        displacement = age * velocity
    elif velocity_decay == 0.0:
        displacement = np.zeros_like(velocity)
    else:
        displacement = velocity * (1.0 - velocity_decay ** age) / (1.0 - velocity_decay)
    return last_box + displacement


def valid_box(box: np.ndarray, width: float, height: float,
              minimum_visible_fraction: float) -> np.ndarray | None:
    if not np.isfinite(box).all():
        return None
    original_width = max(float(box[2] - box[0]), 0.0)
    original_height = max(float(box[3] - box[1]), 0.0)
    original_area = original_width * original_height
    if original_area < 1.0:
        return None
    clipped = box.copy()
    clipped[[0, 2]] = np.clip(clipped[[0, 2]], 0.0, width)
    clipped[[1, 3]] = np.clip(clipped[[1, 3]], 0.0, height)
    visible_area = max(float(clipped[2] - clipped[0]), 0.0) * max(
        float(clipped[3] - clipped[1]), 0.0
    )
    if visible_area / original_area < minimum_visible_fraction:
        return None
    return clipped


def fill_scene(payloads: list[dict], maximum_age: int, maximum_overlap: float,
               velocity_decay: float, minimum_observations: int,
               require_other_view: bool, width: float, height: float,
               minimum_visible_fraction: float,
               geometry_traces: dict | None = None,
               cross_view_weight: float = 0.0,
               velocity_estimator: str = "last",
               timing_samples: list[float] | None = None) -> dict:
    frame_count = min(len(payloads[0]), len(payloads[1]))
    histories = [defaultdict(lambda: deque(maxlen=4)) for _ in range(2)]
    counters = [defaultdict(int), defaultdict(int)]
    for frame in range(frame_count):
        frame_started = time.perf_counter() if timing_samples is not None else None
        frame_key = f"frame={frame}"
        originals = [payloads[view].get(frame_key, []) for view in range(2)]
        observed_ids = [{int(row[0]) for row in rows} for rows in originals]
        observed_rows = [
            {int(row[0]): row for row in rows}
            for rows in originals
        ]
        # Update motion only from real tracker/MIA observations.  Generated
        # boxes never become self-reinforcing state.
        for view in range(2):
            for row in originals[view]:
                identity = int(row[0])
                histories[view][identity].append((
                    frame, np.asarray(row[1:5], dtype=np.float64)
                ))

        for view in range(2):
            rows = payloads[view].setdefault(frame_key, [])
            existing_boxes = np.asarray(
                [row[1:5] for row in rows], dtype=np.float64
            ).reshape(-1, 4)
            candidates = []
            for identity, history in histories[view].items():
                if identity in observed_ids[view]:
                    continue
                age = frame - int(history[-1][0])
                if age < 1 or age > maximum_age:
                    continue
                if len(history) < minimum_observations:
                    counters[view]["insufficient_history_rejected"] += 1
                    continue
                if require_other_view and identity not in observed_ids[1 - view]:
                    counters[view]["other_view_absent_rejected"] += 1
                    continue
                box = predict_box(history, frame, velocity_decay, velocity_estimator)
                if (box is not None and geometry_traces is not None
                        and cross_view_weight > 0.0
                        and identity in observed_rows[1 - view]):
                    trace = geometry_traces.get(frame_key, {})
                    direction = "view2_to_view1" if view == 0 else "view1_to_view2"
                    homography = trace.get(direction)
                    if homography is not None:
                        other_box = np.asarray(
                            observed_rows[1 - view][identity][1:5], dtype=np.float32
                        )
                        other_center = np.asarray([[[
                            0.5 * (other_box[0] + other_box[2]),
                            0.5 * (other_box[1] + other_box[3]),
                        ]]], dtype=np.float32)
                        mapped_center = cv2.perspectiveTransform(
                            other_center, np.asarray(homography, dtype=np.float64)
                        )[0, 0].astype(np.float64)
                        predicted_center = 0.5 * (box[:2] + box[2:])
                        size = box[2:] - box[:2]
                        fused_center = (
                            (1.0 - cross_view_weight) * predicted_center
                            + cross_view_weight * mapped_center
                        )
                        box = np.concatenate([
                            fused_center - 0.5 * size,
                            fused_center + 0.5 * size,
                        ])
                box = valid_box(box, width, height, minimum_visible_fraction)
                if box is None:
                    counters[view]["boundary_rejected"] += 1
                    continue
                candidates.append((age, identity, box))

            used = []
            for age, identity, box in sorted(candidates, key=lambda item: item[0]):
                if len(existing_boxes) and float(iou_matrix(
                        box[None] - 1.0, existing_boxes - 1.0).max()) >= maximum_overlap:
                    counters[view]["overlap_rejected"] += 1
                    continue
                if used and float(iou_matrix(
                        box[None] - 1.0, np.asarray(used) - 1.0).max()) >= maximum_overlap:
                    counters[view]["proposal_competition_rejected"] += 1
                    continue
                rows.append([identity, *box.tolist()])
                used.append(box)
                counters[view]["inserted"] += 1
            counters[view]["proposed"] += len(candidates)
        if timing_samples is not None:
            timing_samples.append(time.perf_counter() - frame_started)
    return {str(view + 1): dict(counters[view]) for view in range(2)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", required=True, type=Path)
    parser.add_argument("--scenes", required=True)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--maximum-age", type=int, default=1)
    parser.add_argument("--maximum-overlap", type=float, default=0.1)
    parser.add_argument("--velocity-decay", type=float, default=0.8)
    parser.add_argument("--velocity-estimator", choices=("last", "median", "regression"),
                        default="last")
    parser.add_argument("--minimum-observations", type=int, default=2)
    parser.add_argument("--require-other-view", action="store_true")
    parser.add_argument("--width", type=float, default=1920.0)
    parser.add_argument("--height", type=float, default=1080.0)
    parser.add_argument("--minimum-visible-fraction", type=float, default=0.5)
    parser.add_argument("--geometry-trace", type=Path)
    parser.add_argument("--cross-view-weight", type=float, default=0.0)
    parser.add_argument("--timing-report", type=Path)
    args = parser.parse_args()
    if args.maximum_age < 1:
        parser.error("--maximum-age must be positive")
    if not 0.0 <= args.velocity_decay <= 1.0:
        parser.error("--velocity-decay must be in [0, 1]")
    if not 0.0 <= args.cross_view_weight <= 1.0:
        parser.error("--cross-view-weight must be in [0, 1]")
    if args.cross_view_weight > 0.0 and args.geometry_trace is None:
        parser.error("--cross-view-weight requires --geometry-trace")

    args.out.mkdir(parents=True, exist_ok=True)
    report = {
        "protocol": "strictly causal forward-only track extension; no future endpoint or retroactive write",
        "parameters": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "scenes": {},
    }
    timing_samples: list[float] | None = [] if args.timing_report else None
    scene_timing = {}
    total_started = time.perf_counter()
    for scene in (value.strip() for value in args.scenes.split(",") if value.strip()):
        scene_started = time.perf_counter()
        sample_start = len(timing_samples) if timing_samples is not None else 0
        payloads = [json.loads(
            (args.results / f"{scene}-{view}.json").read_text(encoding="utf-8")
        ) for view in (1, 2)]
        geometry_traces = (json.loads(
            (args.geometry_trace / f"{scene}.json").read_text(encoding="utf-8")
        ) if args.geometry_trace is not None else None)
        scene_report = fill_scene(
            payloads, args.maximum_age, args.maximum_overlap,
            args.velocity_decay, args.minimum_observations,
            args.require_other_view, args.width, args.height,
            args.minimum_visible_fraction,
            geometry_traces=geometry_traces,
            cross_view_weight=args.cross_view_weight,
            velocity_estimator=args.velocity_estimator,
            timing_samples=timing_samples,
        )
        for view in (1, 2):
            (args.out / f"{scene}-{view}.json").write_text(
                json.dumps(payloads[view - 1]), encoding="utf-8"
            )
        report["scenes"][scene] = scene_report
        if timing_samples is not None:
            values = np.asarray(timing_samples[sample_start:], dtype=np.float64)
            scene_timing[scene] = {
                "frames": int(values.size),
                "processing_seconds": float(values.sum()),
                "end_to_end_seconds": float(time.perf_counter() - scene_started),
                "frame_latency_mean_ms": float(values.mean() * 1000.0),
                "frame_latency_p50_ms": float(np.percentile(values, 50) * 1000.0),
                "frame_latency_p95_ms": float(np.percentile(values, 95) * 1000.0),
            }
        print(f"scene={scene} report={scene_report}", flush=True)
    (args.out / "causal_forward_fill_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    if args.timing_report:
        values = np.asarray(timing_samples, dtype=np.float64)
        processing_seconds = float(values.sum())
        timing_report = {
            "scope": "P2 causal recovery incremental module",
            "parameters": 0,
            "frames": int(values.size),
            "processing_seconds": processing_seconds,
            "end_to_end_seconds": float(time.perf_counter() - total_started),
            "frame_latency_mean_ms": float(values.mean() * 1000.0),
            "frame_latency_p50_ms": float(np.percentile(values, 50) * 1000.0),
            "frame_latency_p95_ms": float(np.percentile(values, 95) * 1000.0),
            "frames_per_second": float(values.size / processing_seconds),
            "per_scene": scene_timing,
        }
        args.timing_report.parent.mkdir(parents=True, exist_ok=True)
        args.timing_report.write_text(json.dumps(timing_report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
