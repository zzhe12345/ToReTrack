from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
import xml.etree.ElementTree as ET



IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


@dataclass(frozen=True)
class ScenePair:
    scene_id: str
    view1_frames: Path
    view2_frames: Path
    view1_xml: Path
    view2_xml: Path


def _scene_id(name: str) -> str | None:
    m = re.match(r"^(\d+)-([12])(?:\.xml)?$", name)
    return m.group(1) if m else None


def _collect_frame_dirs(data_root: Path, view: int) -> dict[str, Path]:
    """Collect one directory per scene/view from flat or split MDMT layouts."""
    flat_view = data_root / str(view)
    candidates = (
        [p for p in flat_view.iterdir() if p.is_dir()]
        if flat_view.is_dir()
        else [p for p in data_root.rglob(f"*-{view}") if p.is_dir()]
    )
    result: dict[str, Path] = {}
    for path in candidates:
        scene_id = _scene_id(path.name)
        if scene_id is None or not path.name.endswith(f"-{view}"):
            continue
        if scene_id in result:
            raise ValueError(
                f"Duplicate frame directories for scene {scene_id}, view {view}: "
                f"{result[scene_id]} and {path}"
            )
        result[scene_id] = path
    return result


def discover_scene_pairs(data_root: str | Path, xml_root: str | Path) -> list[ScenePair]:
    data_root = Path(data_root)
    xml_root = Path(xml_root)
    data1 = _collect_frame_dirs(data_root, 1)
    data2 = _collect_frame_dirs(data_root, 2)
    xml1 = {_scene_id(p.name): p for p in (xml_root / "1").glob("*.xml") if _scene_id(p.name)}
    xml2 = {_scene_id(p.name): p for p in (xml_root / "2").glob("*.xml") if _scene_id(p.name)}
    scene_ids = sorted(set(data1) & set(data2) & set(xml1) & set(xml2), key=lambda x: int(x))
    return [ScenePair(s, data1[s], data2[s], xml1[s], xml2[s]) for s in scene_ids]


def list_frame_files(frame_dir: Path) -> list[Path]:
    return sorted([p for p in frame_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS])


def load_split_scene_ids(split_file: str | Path, split: str) -> list[str]:
    data = json.loads(Path(split_file).read_text(encoding="utf-8"))
    splits = data.get("splits", data)
    if split == "all":
        ids: list[str] = []
        for name in ["train", "val", "test"]:
            ids.extend(str(s) for s in splits.get(name, []))
        return ids
    if split not in splits:
        raise KeyError(f"Split {split!r} not found in {split_file}")
    return [str(s) for s in splits[split]]


def select_scene_pairs(pairs: list[ScenePair], split: str, val_ratio: float = 0.2, split_file: str | Path | None = None) -> list[ScenePair]:
    if split == "all":
        return pairs
    if split_file:
        wanted = set(load_split_scene_ids(split_file, split))
        selected = [pair for pair in pairs if pair.scene_id in wanted]
        if not selected:
            raise FileNotFoundError(f"No scenes from split={split!r} in {split_file} were found in the dataset roots.")
        return selected
    cut = max(1, int(round(len(pairs) * (1.0 - val_ratio))))
    if split == "train":
        return pairs[:cut]
    if split in {"val", "test"}:
        return pairs[cut:] or pairs[-1:]
    return pairs


def parse_cvat_tracks(xml_path: str | Path) -> dict[int, list[dict]]:
    root = ET.parse(xml_path).getroot()
    frames: dict[int, list[dict]] = {}
    for track in root.findall("track"):
        track_id = int(track.attrib["id"])
        label = track.attrib.get("label", "target")
        for box in track.findall("box"):
            outside = int(float(box.attrib.get("outside", "0")))
            if outside:
                continue
            frame = int(box.attrib["frame"])
            xtl = float(box.attrib["xtl"])
            ytl = float(box.attrib["ytl"])
            xbr = float(box.attrib["xbr"])
            ybr = float(box.attrib["ybr"])
            occluded = int(float(box.attrib.get("occluded", "0")))
            frames.setdefault(frame, []).append({
                "track_id": track_id,
                "label": label,
                "bbox": [xtl, ytl, xbr, ybr],
                "occluded": occluded,
                "confidence": 1.0 if not occluded else 0.7,
            })
    return frames
