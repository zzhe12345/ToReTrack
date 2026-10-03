from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

from .dataset import discover_scene_pairs, list_frame_files, select_scene_pairs


def overlapping_tiles(height: int, width: int, grid: int, overlap: float) -> list[tuple[int, int, int, int]]:
    """Return a deterministic overlapping grid covering the full image."""
    grid = max(int(grid), 1)
    overlap = min(max(float(overlap), 0.0), 0.9)
    if grid == 1:
        return [(0, 0, width, height)]

    def starts(length: int) -> tuple[list[int], int]:
        tile = min(length, int(np.ceil(length / (grid - overlap * (grid - 1)))))
        if tile >= length:
            return [0], length
        step = max(int(round(tile * (1.0 - overlap))), 1)
        values = [min(index * step, length - tile) for index in range(grid)]
        values[-1] = length - tile
        return list(dict.fromkeys(values)), tile

    xs, tile_width = starts(width)
    ys, tile_height = starts(height)
    return [(x, y, x + tile_width, y + tile_height) for y in ys for x in xs]


def _load_esod_modules(esod_root: Path):
    """Import the official ESOD repository without installing it as a package."""
    # ESOD registers optional OpenMMLab models during import. Install the same
    # supported operators/cache routing before those imports as the MIA stage.
    from .mmcv_ops_compat import install
    install()
    root = str(esod_root.resolve())
    if root not in sys.path:
        sys.path.insert(0, root)
    from models.experimental import attempt_load  # type: ignore
    from utils.datasets import letterbox, norm_imgs  # type: ignore
    from utils.general import check_img_size, non_max_suppression, scale_coords  # type: ignore

    return attempt_load, letterbox, norm_imgs, check_img_size, non_max_suppression, scale_coords


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Full44 detection caches with official ESOD weights.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--split", choices=("train", "val", "test", "all"), default="val")
    parser.add_argument("--data", default="data")
    parser.add_argument("--xml", default="newxml")
    parser.add_argument("--split-file", default="configs/splits.json")
    parser.add_argument("--esod-root", default="third_party/esod")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--img-size", type=int, default=1280)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--conf-thres", type=float, default=0.01)
    parser.add_argument("--iou-thres", type=float, default=0.50)
    parser.add_argument("--max-det", type=int, default=1000)
    parser.add_argument("--max-frames-per-view", type=int, default=0,
                        help="Optional smoke-test limit; 0 processes every frame.")
    parser.add_argument("--half", action="store_true")
    parser.add_argument("--scene-ids", default="", help="Optional comma-separated scene subset.")
    parser.add_argument("--base-cache-root", default="", help="Reuse and supplement an existing full-frame cache.")
    parser.add_argument("--tile-grid", type=int, default=1)
    parser.add_argument("--tile-overlap", type=float, default=0.20)
    parser.add_argument("--tile-score-scale", type=float, default=1.0)
    parser.add_argument("--merge-iou", type=float, default=0.60)
    args = parser.parse_args()

    esod_root = Path(args.esod_root)
    attempt_load, letterbox, norm_imgs, check_img_size, non_max_suppression, scale_coords = (
        _load_esod_modules(esod_root)
    )

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    checkpoint = Path(args.checkpoint).resolve()
    model = attempt_load(str(checkpoint), map_location=device).eval()
    stride = int(model.stride.max())
    image_size = int(check_img_size(args.img_size, s=stride))
    use_half = bool(args.half and device.type == "cuda")
    model.half() if use_half else model.float()

    pairs = select_scene_pairs(
        discover_scene_pairs(args.data, args.xml), args.split, split_file=args.split_file,
    )
    if args.scene_ids:
        wanted = {value.strip() for value in args.scene_ids.split(",") if value.strip()}
        pairs = [pair for pair in pairs if str(pair.scene_id) in wanted]
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    report: dict = {
        "architecture": "ESOD",
        "checkpoint": str(checkpoint),
        "split": args.split,
        "img_size": image_size,
        "conf_thres": args.conf_thres,
        "iou_thres": args.iou_thres,
        "base_cache_root": args.base_cache_root,
        "tile_grid": args.tile_grid,
        "tile_overlap": args.tile_overlap,
        "tile_score_scale": args.tile_score_scale,
        "merge_iou": args.merge_iou,
        "files": {},
    }

    # Warm-up uses the same tensor geometry as real batches.
    warmup = torch.zeros(1, 3, image_size, image_size, device=device)
    warmup = warmup.half() if use_half else warmup.float()
    model(warmup)

    def infer_images(images: list[np.ndarray]) -> list[tuple[np.ndarray, np.ndarray]]:
        results: list[tuple[np.ndarray, np.ndarray]] = []
        for start in range(0, len(images), args.batch_size):
            originals = images[start:start + args.batch_size]
            network_images = []
            for image in originals:
                resized = letterbox(image, image_size, stride=stride)[0]
                resized = resized[:, :, ::-1].transpose(2, 0, 1)
                network_images.append(np.ascontiguousarray(resized))
            tensor = torch.from_numpy(np.stack(network_images)).to(device)
            tensor = tensor.half() if use_half else tensor.float()
            tensor = norm_imgs(tensor, model)
            prediction = model(tensor)[0][0]
            detections = non_max_suppression(
                prediction, args.conf_thres, args.iou_thres,
                agnostic=True, max_det=args.max_det,
            )
            for det, image in zip(detections, originals):
                if len(det):
                    det[:, :4] = scale_coords(tensor.shape[2:], det[:, :4], image.shape)
                    results.append((
                        det[:, :4].float().cpu().numpy().astype(np.float32, copy=False),
                        det[:, 4].float().cpu().numpy().astype(np.float32, copy=False),
                    ))
                else:
                    results.append((np.empty((0, 4), np.float32), np.empty((0,), np.float32)))
        return results

    for pair in pairs:
        for view_id, frame_dir in ((1, pair.view1_frames), (2, pair.view2_frames)):
            frames = list_frame_files(frame_dir)
            if args.max_frames_per_view > 0:
                frames = frames[:args.max_frames_per_view]
            cached: list[dict] = []
            total = 0
            if args.tile_grid > 1:
                if not args.base_cache_root:
                    raise ValueError("--tile-grid > 1 requires --base-cache-root for full+tile fusion")
                base_path = Path(args.base_cache_root) / f"scene{pair.scene_id}_view{view_id}.pt"
                base_frames = torch.load(base_path, map_location="cpu", weights_only=False)
                if len(base_frames) < len(frames):
                    raise ValueError(f"Base cache is shorter than source frames: {base_path}")
                from torchvision.ops import nms
                for frame_index, path in enumerate(frames):
                    image = cv2.imread(str(path))
                    if image is None:
                        raise FileNotFoundError(path)
                    tiles = overlapping_tiles(image.shape[0], image.shape[1], args.tile_grid, args.tile_overlap)
                    tile_images = [image[y1:y2, x1:x2] for x1, y1, x2, y2 in tiles]
                    tile_predictions = infer_images(tile_images)
                    box_parts = [np.asarray(base_frames[frame_index]["boxes"], np.float32).reshape(-1, 4)]
                    score_parts = [np.asarray(base_frames[frame_index]["scores"], np.float32).reshape(-1)]
                    for (tile_boxes, tile_scores), (x1, y1, _, _) in zip(tile_predictions, tiles):
                        mapped = tile_boxes.copy()
                        if len(mapped):
                            mapped[:, [0, 2]] += x1
                            mapped[:, [1, 3]] += y1
                        box_parts.append(mapped)
                        score_parts.append(tile_scores * float(args.tile_score_scale))
                    boxes = np.concatenate(box_parts) if box_parts else np.empty((0, 4), np.float32)
                    scores = np.concatenate(score_parts) if score_parts else np.empty((0,), np.float32)
                    if len(boxes):
                        keep = nms(torch.from_numpy(boxes), torch.from_numpy(scores), args.merge_iou)
                        keep = keep[:args.max_det].cpu().numpy()
                        boxes, scores = boxes[keep], scores[keep]
                    cached.append({"boxes": boxes, "scores": scores, "embeddings": None})
                    total += len(boxes)
                    done = frame_index + 1
                    if frame_index == 0 or done % 100 == 0 or done == len(frames):
                        print(
                            f"scene={pair.scene_id} view={view_id} frame={done}/{len(frames)} "
                            f"detections={total} tiles={len(tiles)}", flush=True,
                        )
                name = f"scene{pair.scene_id}_view{view_id}.pt"
                torch.save(cached, out_root / name)
                report["files"][name] = {"frames": len(frames), "detections": total}
                (out_root / "esod_cache_report.json").write_text(
                    json.dumps(report, indent=2), encoding="utf-8",
                )
                continue
            for start in range(0, len(frames), args.batch_size):
                paths = frames[start:start + args.batch_size]
                original_images = []
                network_images = []
                for path in paths:
                    image = cv2.imread(str(path))
                    if image is None:
                        raise FileNotFoundError(path)
                    original_images.append(image)
                    resized = letterbox(image, image_size, stride=stride)[0]
                    resized = resized[:, :, ::-1].transpose(2, 0, 1)
                    network_images.append(np.ascontiguousarray(resized))

                tensor = torch.from_numpy(np.stack(network_images)).to(device)
                tensor = tensor.half() if use_half else tensor.float()
                tensor = norm_imgs(tensor, model)
                output = model(tensor)
                prediction = output[0][0]
                detections = non_max_suppression(
                    prediction,
                    args.conf_thres,
                    args.iou_thres,
                    agnostic=True,
                    max_det=args.max_det,
                )

                for det, image in zip(detections, original_images):
                    if len(det):
                        det[:, :4] = scale_coords(tensor.shape[2:], det[:, :4], image.shape).round()
                        boxes = det[:, :4].float().cpu().numpy().astype(np.float32, copy=False)
                        scores = det[:, 4].float().cpu().numpy().astype(np.float32, copy=False)
                    else:
                        boxes = np.empty((0, 4), dtype=np.float32)
                        scores = np.empty((0,), dtype=np.float32)
                    cached.append({"boxes": boxes, "scores": scores, "embeddings": None})
                    total += len(boxes)

                done = min(start + len(paths), len(frames))
                if start == 0 or done % 100 == 0 or done == len(frames):
                    print(
                        f"scene={pair.scene_id} view={view_id} frame={done}/{len(frames)} "
                        f"detections={total}",
                        flush=True,
                    )

            name = f"scene{pair.scene_id}_view{view_id}.pt"
            torch.save(cached, out_root / name)
            report["files"][name] = {"frames": len(frames), "detections": total}
            (out_root / "esod_cache_report.json").write_text(
                json.dumps(report, indent=2), encoding="utf-8",
            )


if __name__ == "__main__":
    main()
