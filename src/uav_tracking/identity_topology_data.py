from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import torch

from .dataset import parse_cvat_tracks


@dataclass(frozen=True)
class TopologyWindow:
    scene_id: str
    start: int
    left_indices: tuple[int, ...]
    right_indices: tuple[int, ...]
    positive_pairs: tuple[tuple[int, int], ...]


def _records_by_track(xml_path: Path) -> dict[int, list[tuple[int, dict]]]:
    by_track: dict[int, list[tuple[int, dict]]] = defaultdict(list)
    for frame, records in parse_cvat_tracks(xml_path).items():
        for record in records:
            by_track[int(record["track_id"])].append((int(frame), record))
    return by_track


def trajectory_features(
    records: list[tuple[int, dict]], start: int, window_size: int, width: float, height: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the paper's current state and one-step trajectory state.

    A topology node is represented at its latest valid time in the current
    tracklet window.  Motion is the normalized box-state difference to the
    immediately preceding frame, followed by the binary history-valid flag.
    This distinguishes a missing predecessor from a genuine zero displacement.
    """
    # Official supplement output very rarely retains two rows with one ID in
    # the same frame.  A physical track has one state at t; match the FPN path
    # by deterministically retaining the first detector row for that frame.
    unique = {}
    for frame, record in records:
        if start <= frame < start + window_size and frame not in unique:
            unique[frame] = record
    rows = list(unique.items())
    if not rows:
        return torch.zeros(4), torch.zeros(5)
    rows.sort(key=lambda item: item[0])
    boxes = torch.tensor([record["bbox"] for _, record in rows], dtype=torch.float32)
    state = torch.stack(
        [(boxes[:, 0] + boxes[:, 2]) * 0.5 / width, (boxes[:, 1] + boxes[:, 3]) * 0.5 / height,
         (boxes[:, 2] - boxes[:, 0]).clamp_min(1.0) / width,
         (boxes[:, 3] - boxes[:, 1]).clamp_min(1.0) / height], dim=1
    )
    spatial = state[-1]
    valid = len(state) >= 2 and rows[-2][0] == rows[-1][0] - 1
    delta = state[-1] - state[-2] if valid else state.new_zeros(4)
    trajectory = torch.cat([delta, state.new_tensor([float(valid)])])
    return spatial, trajectory


class TopologyFeatureDataset:
    """Adapter from a frozen ReID window cache to complete topology samples."""

    def __init__(
        self,
        cache_path: str | Path,
        xml_root: str | Path,
        prototype: str = "mean",
        window_size: int = 16,
        image_size: tuple[float, float] = (1920.0, 1080.0),
    ) -> None:
        payload = torch.load(cache_path, map_location="cpu", weights_only=False)
        self.appearance_metadata = payload.get("metadata", {})
        self.appearance = payload["prototypes"][prototype].float()
        frame_data = payload.get("frame_data", [])
        if "mean_qualities" in payload:
            self.quality = payload["mean_qualities"].float()
        else:
            self.quality = torch.tensor([
                float(row.get("qualities", torch.ones(1)).float().mean())
                for row in frame_data
            ]) if frame_data else torch.ones(len(self.appearance))
        keys = payload["metadata"]["item_keys"]
        if len(keys) != len(self.appearance):
            raise ValueError("cache item keys and prototypes have different lengths")
        self.keys = [(str(scene), int(view), int(start), int(track)) for scene, view, start, track in keys]
        self.window_size = window_size
        self.spatial = torch.zeros(len(keys), 4)
        self.trajectory = torch.zeros(len(keys), 5)
        self.occlusion = torch.zeros(len(keys))
        grouped: dict[tuple[str, int], list[int]] = defaultdict(list)
        for index, (scene, view, start, _) in enumerate(self.keys):
            grouped[(scene, start)].append(index)

        xml_root = Path(xml_root)
        track_cache: dict[tuple[str, int], dict[int, list[tuple[int, dict]]]] = {}
        width, height = image_size
        for index, (scene, view, start, track) in enumerate(self.keys):
            cache_key = (scene, view)
            if cache_key not in track_cache:
                track_cache[cache_key] = _records_by_track(xml_root / str(view) / f"{scene}-{view}.xml")
            spatial, trajectory = trajectory_features(
                track_cache[cache_key].get(track, []), start, window_size, width, height
            )
            self.spatial[index] = spatial
            self.trajectory[index] = trajectory
            local_rows = [record for frame, record in track_cache[cache_key].get(track, [])
                          if start <= frame < start + window_size]
            self.occlusion[index] = (sum(bool(row.get("occluded", 0)) for row in local_rows)
                                     / max(len(local_rows), 1))

        windows: list[TopologyWindow] = []
        for (scene, start), indices in sorted(grouped.items(), key=lambda item: (int(item[0][0]), item[0][1])):
            left = [index for index in indices if self.keys[index][1] == 1]
            right = [index for index in indices if self.keys[index][1] == 2]
            if not left or not right:
                continue
            right_lookup = {self.keys[index][3]: column for column, index in enumerate(right)}
            positives = [
                (row, right_lookup[self.keys[index][3]]) for row, index in enumerate(left)
                if self.keys[index][3] in right_lookup
            ]
            if positives:
                windows.append(TopologyWindow(scene, start, tuple(left), tuple(right), tuple(positives)))
        self.windows = windows

    @property
    def appearance_dim(self) -> int:
        return int(self.appearance.shape[1])

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, index: int) -> dict:
        window = self.windows[index]
        left = torch.tensor(window.left_indices, dtype=torch.long)
        right = torch.tensor(window.right_indices, dtype=torch.long)
        pairs = torch.tensor(window.positive_pairs, dtype=torch.long)
        return {
            "scene_id": window.scene_id,
            "start": window.start,
            "left_keys": [self.keys[index] for index in window.left_indices],
            "right_keys": [self.keys[index] for index in window.right_indices],
            "left": (self.appearance[left], self.spatial[left], self.trajectory[left]),
            "right": (self.appearance[right], self.spatial[right], self.trajectory[right]),
            "positive_pairs": pairs,
            "mean_quality": float(torch.cat([
                self.quality[left], self.quality[right]
            ]).mean()),
            "mean_occlusion": float(torch.cat([
                self.occlusion[left], self.occlusion[right]
            ]).mean()),
        }


class PredictedTopologyFeatureDataset(TopologyFeatureDataset):
    """Topology samples built from detector/tracker outputs, not GT boxes.

    Ground truth IDs are used only to construct supervision/evaluation pairs;
    all node features are computed from predicted tracklet observations.
    """

    def __init__(self, cache_path: str | Path, prediction_path: str | Path,
                 prototype: str = "mean", window_size: int = 16,
                 image_size: tuple[float, float] = (1920.0, 1080.0)) -> None:
        cache = torch.load(cache_path, map_location="cpu", weights_only=False)
        self.appearance_metadata = cache.get("metadata", {})
        prediction = torch.load(prediction_path, map_location="cpu", weights_only=False)
        self.prediction_metadata = {
            "report": prediction.get("report", {}),
            "args": prediction.get("args", {}),
            "representation": prediction.get("representation"),
        }
        self.appearance = cache["prototypes"][prototype].float()
        self.keys = [tuple(item.key) for item in prediction["items"]]
        cache_keys = [tuple(key) for key in cache["metadata"]["item_keys"]]
        if cache_keys != self.keys or len(self.appearance) != len(self.keys):
            raise ValueError("prediction items and embedding cache are not aligned")
        self.window_size = window_size
        self.quality = cache.get("mean_qualities", torch.ones(len(self.keys))).float()
        if "mean_qualities" not in cache and cache.get("frame_data"):
            self.quality = torch.tensor([
                float(row.get("qualities", torch.ones(1)).float().mean())
                for row in cache["frame_data"]
            ])
        self.spatial = torch.zeros(len(self.keys), 4)
        self.trajectory = torch.zeros(len(self.keys), 5)
        self.occlusion = torch.zeros(len(self.keys))
        grouped: dict[tuple[str, int], list[int]] = defaultdict(list)
        width, height = image_size
        xml_root = Path(prediction.get("args", {}).get("xml", "data/datafull/new_xml"))
        gt_track_cache = {}
        for index, item in enumerate(prediction["items"]):
            records = [(int(obs.frame), {"bbox": obs.bbox}) for obs in item.observations]
            spatial, trajectory = trajectory_features(
                records, int(item.start), window_size, width, height
            )
            self.spatial[index] = spatial; self.trajectory[index] = trajectory
            grouped[(str(item.scene_id), int(item.start))].append(index)
            label = prediction.get("truth", {}).get(index, {}).get("gt_id")
            if label is not None:
                cache_key = (str(item.scene_id), int(item.view_id))
                if cache_key not in gt_track_cache:
                    xml_path = xml_root / str(item.view_id) / f"{item.scene_id}-{item.view_id}.xml"
                    gt_track_cache[cache_key] = (_records_by_track(xml_path) if xml_path.exists() else {})
                frames = {int(obs.frame) for obs in item.observations}
                gt_rows = [record for frame, record in gt_track_cache[cache_key].get(int(label), [])
                           if frame in frames]
                self.occlusion[index] = (sum(bool(row.get("occluded", 0)) for row in gt_rows)
                                         / max(len(gt_rows), 1))

        truth = prediction.get("truth", {})
        windows = []
        for (scene, start), indices in sorted(grouped.items(), key=lambda item: (int(item[0][0]), item[0][1])):
            left = [index for index in indices if int(self.keys[index][1]) == 1]
            right = [index for index in indices if int(self.keys[index][1]) == 2]
            if not left or not right:
                continue
            best_left, best_right = {}, {}
            for rows, target in ((left, best_left), (right, best_right)):
                for index in rows:
                    label = truth.get(index, {}).get("gt_id")
                    if label is None:
                        continue
                    score = (float(truth[index].get("purity", 0.0)),
                             float(truth[index].get("tracking_quality", 0.0)))
                    if label not in target or score > target[label][0]:
                        target[label] = (score, index)
            left_column = {index: row for row, index in enumerate(left)}
            right_column = {index: col for col, index in enumerate(right)}
            positives = [
                (left_column[best_left[label][1]], right_column[best_right[label][1]])
                for label in sorted(set(best_left) & set(best_right))
            ]
            if positives:
                windows.append(TopologyWindow(scene, start, tuple(left), tuple(right), tuple(positives)))
        self.windows = windows


class SynchronousMIAFrameDataset:
    """Paper-aligned samples whose two views contain targets at one frame t."""

    def __init__(self, cache_path: str | Path | None = None, *, payload: dict | None = None) -> None:
        if payload is None:
            if cache_path is None:
                raise ValueError("cache_path or payload is required")
            payload = torch.load(cache_path, map_location="cpu", weights_only=False)
        self.metadata = payload.get("metadata", {})
        if self.metadata.get("representation") != "mia_net_synchronous_frame_targets":
            raise ValueError("cache is not a synchronous MIA frame representation")
        self.samples = payload.get("samples", [])
        if not self.samples:
            raise ValueError("synchronous frame cache contains no samples")
        for sample in self.samples:
            sample["left"] = tuple(value.float() for value in sample["left"])
            sample["right"] = tuple(value.float() for value in sample["right"])
        self.appearance_metadata = self.metadata
        self.prediction_metadata = {
            "report": {"frontend": self.metadata.get("frontend")},
            "args": {"split": self.metadata.get("split")},
            "representation": self.metadata.get("representation"),
        }

    @property
    def appearance_dim(self) -> int:
        for sample in self.samples:
            for side in ("left", "right"):
                if len(sample[side][0]):
                    return int(sample[side][0].shape[1])
        raise ValueError("synchronous frame cache contains no target appearance")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict:
        return self.samples[index]
