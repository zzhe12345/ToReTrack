"""Validate the synchronous topology feature contract."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


EXPECTED = {
    "representation": "mia_net_synchronous_frame_targets",
    "appearance_source": "frozen_official_mia_detector_fpn",
    "temporal_aggregation": "current_frame_only",
    "topology_time": "all nodes are targets observed in the same synchronized frame t",
    "trajectory": "normalized box state at t minus t-1 plus binary validity",
    "geometry_direction": "ordered bidirectional view1_to_view2 and view2_to_view1",
    "geometry_score": "exp(-directed_mapped_center_distance/state_scale)",
    "geometry_scales": {"new": 80.0, "existing": 50.0},
    "geometry_state": "official MIA new iff current ID exceeds previous committed maximum ID",
    "candidate_rule": "all synchronized target pairs; geometry is soft and never deletes candidates",
    "geometry_reachability": "diagnostic only; never used as a candidate mask",
}


def audit(path: Path, expected_insertion_point: str = "",
          expected_pooling: str = "") -> dict:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    metadata = payload.get("metadata", {})
    for key, expected in EXPECTED.items():
        if metadata.get(key) != expected:
            raise ValueError(f"{path}: metadata {key}={metadata.get(key)!r}, expected {expected!r}")
    if metadata.get("frontend") != "autoassign_bytetrack":
        raise ValueError(f"{path}: formal branch requires AutoAssign+ByteTrack")
    # User-trained detector files have portable paths. Validate provenance is
    # present rather than requiring the author's old absolute weight location.
    if not metadata.get("detector_checkpoint"):
        raise ValueError(f"{path}: missing AutoAssign checkpoint provenance")
    insertion_point = metadata.get("insertion_point", "post_mia_output")
    pooling = metadata.get("pooling")
    allowed_pooling = {
        "per_level_roi_align_1x1_then_equal_fpn_mean": 256,
        "assigned_fpn_roi_align_3x3_meanmax": 512,
    }
    if insertion_point not in {"post_mia_output", "pre_cross_view_id_allocation"}:
        raise ValueError(f"{path}: unknown insertion point {insertion_point!r}")
    if pooling not in allowed_pooling:
        raise ValueError(f"{path}: unknown pooling {pooling!r}")
    if expected_insertion_point and insertion_point != expected_insertion_point:
        raise ValueError(f"{path}: insertion point {insertion_point!r} != {expected_insertion_point!r}")
    if expected_pooling and pooling != expected_pooling:
        raise ValueError(f"{path}: pooling {pooling!r} != {expected_pooling!r}")
    appearance_dim = allowed_pooling[pooling]

    samples = payload.get("samples", [])
    if not samples:
        raise ValueError(f"{path}: empty sample list")
    previous = None
    scenes = set()
    nodes = positives = candidates = 0
    same_id_positives = repair_positives = repair_candidates = 0
    for index, sample in enumerate(samples):
        scene, frame = str(sample["scene_id"]), int(sample["frame"])
        order = (int(scene), frame)
        if previous is not None and order < previous:
            raise ValueError(f"{path}: samples are not chronological at index {index}")
        previous = order
        scenes.add(scene)
        lengths = []
        for view, side in ((1, "left"), (2, "right")):
            keys = sample[f"{side}_keys"]
            appearance, spatial, trajectory = sample[side]
            count = len(keys)
            lengths.append(count)
            if (appearance.shape != (count, appearance_dim) or spatial.shape != (count, 4)
                    or trajectory.shape != (count, 5)):
                raise ValueError(f"{path}: malformed {side} tensors at index {index}")
            if len(set(tuple(key) for key in keys)) != count:
                raise ValueError(f"{path}: duplicate {side} node key at index {index}")
            if any(str(key[0]) != scene or int(key[1]) != view or int(key[2]) != frame for key in keys):
                raise ValueError(f"{path}: asynchronous {side} key at index {index}")
            for tensor in (appearance, spatial, trajectory):
                if not torch.isfinite(tensor).all():
                    raise ValueError(f"{path}: non-finite {side} feature at index {index}")
            if count:
                norms = appearance.float().norm(dim=1)
                if not torch.allclose(norms, torch.ones_like(norms), atol=3e-3, rtol=3e-3):
                    raise ValueError(f"{path}: non-unit official FPN feature at index {index}")
                valid = trajectory[:, 4]
                if not torch.all((valid == 0) | (valid == 1)):
                    raise ValueError(f"{path}: non-binary trajectory validity at index {index}")
                if not torch.allclose(trajectory[valid == 0, :4], torch.zeros_like(trajectory[valid == 0, :4])):
                    raise ValueError(f"{path}: invalid history has nonzero motion at index {index}")
            nodes += count
        nl, nr = lengths
        geometry, mask = sample["geometric_score"], sample["candidate_mask"]
        reverse_geometry = sample.get("reverse_geometric_score")
        reverse_mask = sample.get("reverse_candidate_mask")
        if geometry.shape != (nl, nr) or mask.shape != (nl, nr):
            raise ValueError(f"{path}: geometry shape mismatch at index {index}")
        if reverse_geometry is None or reverse_geometry.shape != (nr, nl):
            raise ValueError(f"{path}: reverse geometry shape mismatch at index {index}")
        if reverse_mask is None or reverse_mask.shape != (nr, nl):
            raise ValueError(f"{path}: reverse candidate shape mismatch at index {index}")
        if not torch.isfinite(geometry).all() or bool(((geometry < 0) | (geometry > 1)).any()):
            raise ValueError(f"{path}: invalid soft geometry at index {index}")
        if (not torch.isfinite(reverse_geometry).all()
                or bool(((reverse_geometry < 0) | (reverse_geometry > 1)).any())):
            raise ValueError(f"{path}: invalid reverse soft geometry at index {index}")
        if nl and nr and not bool(mask.all()):
            raise ValueError(f"{path}: forward geometry still hard-deletes candidates at index {index}")
        if nl and nr and not bool(reverse_mask.all()):
            raise ValueError(f"{path}: reverse geometry still hard-deletes candidates at index {index}")
        new_target = sample.get("left_new_target")
        if new_target is None or new_target.shape != (nl,) or new_target.dtype != torch.bool:
            raise ValueError(f"{path}: missing explicit MIA new/existing state at index {index}")
        new_target = sample.get("right_new_target")
        if new_target is None or new_target.shape != (nr,) or new_target.dtype != torch.bool:
            raise ValueError(f"{path}: missing reverse MIA new/existing state at index {index}")
        pairs = sample["positive_pairs"]
        if pairs.ndim != 2 or pairs.shape[1] != 2:
            raise ValueError(f"{path}: malformed supervision pairs at index {index}")
        if len(pairs):
            if (int(pairs[:, 0].min()) < 0 or int(pairs[:, 0].max()) >= nl
                    or int(pairs[:, 1].min()) < 0 or int(pairs[:, 1].max()) >= nr):
                raise ValueError(f"{path}: supervision pair out of bounds at index {index}")
            if len(set(map(tuple, pairs.tolist()))) != len(pairs):
                raise ValueError(f"{path}: duplicate supervision pair at index {index}")
            if len(set(pairs[:, 0].tolist())) != len(pairs) or len(set(pairs[:, 1].tolist())) != len(pairs):
                raise ValueError(f"{path}: supervision is not one-to-one at index {index}")
        positives += len(pairs)
        candidates += int(mask.sum())
        if nl and nr:
            left_ids = torch.as_tensor([int(key[3]) for key in sample["left_keys"]])
            right_ids = torch.as_tensor([int(key[3]) for key in sample["right_keys"]])
            different_id = left_ids[:, None] != right_ids[None, :]
            repair_candidates += int((mask & reverse_mask.T & different_id).sum())
            if len(pairs):
                repair = different_id[pairs[:, 0], pairs[:, 1]]
                repair_positives += int(repair.sum())
                same_id_positives += int((~repair).sum())
    return {
        "cache": str(path.resolve()), "split": metadata.get("split"),
        "frames": len(samples), "scenes": sorted(scenes, key=int),
        "nodes": nodes, "positive_pairs": positives, "candidate_pairs": candidates,
        "same_id_positive_pairs": same_id_positives,
        "repair_positive_pairs": repair_positives,
        "repair_candidate_pairs": repair_candidates,
        "truth_labels_present": bool(metadata.get("truth_labels_present")),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", action="append", required=True, type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--expected-insertion-point", default="")
    parser.add_argument("--expected-pooling", default="")
    args = parser.parse_args()
    reports = [
        audit(path, args.expected_insertion_point, args.expected_pooling)
        for path in args.cache
    ]
    scene_sets = [set(report["scenes"]) for report in reports]
    for i in range(len(scene_sets)):
        for j in range(i + 1, len(scene_sets)):
            overlap = scene_sets[i] & scene_sets[j]
            if overlap:
                raise ValueError(f"split leakage between caches: {sorted(overlap, key=int)}")
    result = {"status": "PASS", "caches": reports, "split_scene_overlap": 0}
    encoded = json.dumps(result, indent=2, ensure_ascii=False)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(encoded, encoding="utf-8")
    print(encoded)


if __name__ == "__main__":
    main()
