"""Export CVAT split annotations to the MIA/MOT evaluator text format."""
from __future__ import annotations

import argparse
from pathlib import Path

from uav_tracking.dataset import discover_scene_pairs, parse_cvat_tracks, select_scene_pairs


def write_sequence(xml_path: Path, out: Path, frame_offset: int = 0, max_frames: int = 0) -> None:
    """MOT uses frame 0; the released MDA reader subtracts 1 from its input."""
    lines = []
    for frame, records in sorted(parse_cvat_tracks(xml_path).items()):
        if max_frames and frame >= max_frames:
            continue
        for row in records:
            x1, y1, x2, y2 = row["bbox"]
            lines.append(
                f"{frame + frame_offset},{int(row['track_id']) + 1},{x1:g},{y1:g},{x2 - x1:g},{y2 - y1:g},1,1,1"
            )
    out.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", required=True, choices=("train", "val", "test"))
    parser.add_argument("--data", default="data/datafull")
    parser.add_argument("--xml", default="data/datafull/new_xml")
    parser.add_argument("--split-file", default="configs/splits.json")
    parser.add_argument("--scenes", default="", help="Optional comma-separated split subset")
    parser.add_argument("--frame-offset", type=int, choices=(0, 1), default=0)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    pairs = select_scene_pairs(
        discover_scene_pairs(Path(args.data) / args.split, args.xml), args.split,
        split_file=args.split_file,
    )
    args.out.mkdir(parents=True, exist_ok=True)
    if args.scenes:
        wanted = {value.strip() for value in args.scenes.split(",") if value.strip()}
        pairs = [pair for pair in pairs if pair.scene_id in wanted]
    for pair in pairs:
        write_sequence(pair.view1_xml, args.out / f"{pair.scene_id}-1.txt", args.frame_offset, args.max_frames)
        write_sequence(pair.view2_xml, args.out / f"{pair.scene_id}-2.txt", args.frame_offset, args.max_frames)
    print(f"saved MOT ground truth for {len(pairs)} {args.split} scenes to {args.out}")


if __name__ == "__main__":
    main()
