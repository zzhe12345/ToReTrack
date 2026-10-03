"""Convert external CVAT train/val labels into ESOD and AutoAssign inputs."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from uav_tracking.dataset import discover_scene_pairs, list_frame_files, parse_cvat_tracks, select_scene_pairs

CLASS_NAMES = ("pedestrian", "bicycle", "car")
CLASS_ALIASES = {"person": "pedestrian", "people": "pedestrian", "bike": "bicycle", "vehicle": "car"}


def convert(data: Path, output: Path, max_frames: int = 0) -> None:
    output.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val"):
        pairs = select_scene_pairs(discover_scene_pairs(data / split, data / "new_xml"),
                                   split, split_file=ROOT / "configs/splits.json")
        coco = {"images": [], "annotations": [], "categories": [
            {"id": i + 1, "name": name} for i, name in enumerate(CLASS_NAMES)
        ]}
        image_paths = []
        for pair in pairs:
            for view in (1, 2):
                frame_dir, xml = ((pair.view1_frames, pair.view1_xml) if view == 1
                                  else (pair.view2_frames, pair.view2_xml))
                truths = parse_cvat_tracks(xml)
                frames = list_frame_files(frame_dir)
                if max_frames:
                    frames = frames[:max_frames]
                from PIL import Image
                for frame, source in enumerate(frames):
                    with Image.open(source) as image:
                        width, height = image.size
                    name = f"{pair.scene_id}_{view}_{frame:06d}{source.suffix.lower()}"
                    destination = output / "images" / split / name
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    if not destination.exists():
                        try:
                            os.link(source, destination)
                        except OSError:
                            shutil.copy2(source, destination)
                    image_paths.append(destination.as_posix())
                    image_id = len(coco["images"]) + 1
                    coco["images"].append({"id": image_id, "file_name": str(source.resolve()),
                                           "width": width, "height": height})
                    yolo = []
                    for row in truths.get(frame, []):
                        x1, y1, x2, y2 = row["bbox"]
                        x1, x2 = max(0, x1), min(width, x2)
                        y1, y2 = max(0, y1), min(height, y2)
                        if x2 <= x1 or y2 <= y1:
                            continue
                        label = CLASS_ALIASES.get(row["label"].lower(), row["label"].lower())
                        if label not in CLASS_NAMES:
                            raise ValueError(f"Unsupported class {row['label']!r} in {xml}")
                        # ESOD is the single-class proposal detector; AutoAssign
                        # retains the original pedestrian/bicycle/car labels.
                        yolo.append(f"0 {(x1+x2)/2/width:.8f} {(y1+y2)/2/height:.8f} "
                                    f"{(x2-x1)/width:.8f} {(y2-y1)/height:.8f}")
                        coco["annotations"].append({
                            "id": len(coco["annotations"]) + 1, "image_id": image_id,
                            "category_id": CLASS_NAMES.index(label) + 1,
                            "bbox": [x1, y1, x2-x1, y2-y1], "area": (x2-x1)*(y2-y1), "iscrowd": 0,
                        })
                    labels = output / "labels" / split / (Path(name).stem + ".txt")
                    labels.parent.mkdir(parents=True, exist_ok=True)
                    labels.write_text("\n".join(yolo) + ("\n" if yolo else ""), encoding="utf-8")
        if not image_paths:
            raise FileNotFoundError(f"No {split} images found under {data}")
        (output / f"{split}.txt").write_text("\n".join(image_paths) + "\n", encoding="utf-8")
        (output / f"{split}_coco.json").write_text(json.dumps(coco), encoding="utf-8")
        print(f"Converted {split}: {len(image_paths)} images, {len(coco['annotations'])} boxes")
    # The vendored loader creates Gaussian heatmap supervision from augmented
    # boxes on demand, so no dense mask dataset needs to be distributed.
    import yaml
    (output / "esod.yaml").write_text(yaml.safe_dump({
        "train": str(output / "train.txt"), "val": str(output / "val.txt"),
        "nc": 1, "names": ["target"],
    }), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-frames", type=int, default=0)
    args = parser.parse_args()
    convert(args.data.resolve(), args.out.resolve(), args.max_frames)
