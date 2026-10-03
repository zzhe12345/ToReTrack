"""Apply the trained topology branch to official MIA JSON tracking results.

Boxes and within-view track identities are left untouched.  Only cross-view ID
links are replaced by validation-frozen topology-enhanced assignments.  Sparse
MIA tracks that never form a topology window retain their official MIA link.
"""

from __future__ import annotations

import argparse
from collections import defaultdict, deque
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from uav_tracking.identity_links import UnionFind, persistent_cross_view_matches, track_token
from evaluate_identity_topology import collect, load_model, validate_paper_protocol
from train_identity_topology import remember_semantics, temporal_semantics
from uav_tracking.identity_topology_data import (
    PredictedTopologyFeatureDataset, SynchronousMIAFrameDataset,
)
from uav_tracking.repair_gate import (
    candidate_branch_agreement,
    candidate_feature,
    bidirectional_topk_candidates,
    either_direction_topk_candidates,
    load_repair_gate,
    mutual_candidates,
    update_pair_multiframe_evidence,
    update_track_ages,
)


def load_scene_rows(root: Path, scene: str, view: int):
    return json.loads((root / f"{scene}-{view}.json").read_text(encoding="utf-8"))


def ids_in_payload(payload) -> set[int]:
    return {int(row[0]) for rows in payload.values() for row in rows}


def row_tokens(scene: str, view: int, rows: list[list[float]]) -> list[tuple]:
    occurrences = defaultdict(int)
    result = []
    for row in rows:
        identity = int(row[0])
        occurrence = occurrences[identity]
        occurrences[identity] += 1
        result.append((scene, view, identity, occurrence))
    return result


@torch.inference_mode()
def collect_gated_records(model, dataset, gate_bundle, device: str,
                          history_weight: float, history_size: int,
                          candidate_topk: int,
                          candidate_mode: str,
                          gate_threshold: float, minimum_score: float,
                          minimum_margin: float, minimum_support: int):
    """Collect repair links with the exact causal gate feature protocol."""
    gate, feature_mean, feature_scale, _payload = gate_bundle
    model.eval(); gate.eval()
    semantic_memory = defaultdict(lambda: deque(maxlen=history_size))
    age_state = {}
    pair_evidence_state = {}
    records = []
    proposed = gate_passed = policy_passed = 0
    for index in range(len(dataset)):
        sample = dataset[index]
        left = tuple(value.to(device) for value in sample["left"])
        right = tuple(value.to(device) for value in sample["right"])
        geometry = sample["geometric_score"].to(device)
        reverse_geometry = sample["reverse_geometric_score"].to(device)
        _, left_output, right_output = model(left, right, geometry)
        current_topology = 0.5 * (model.topology_score(left_output, right_output) + 1.0)
        left_temporal = temporal_semantics(
            left_output, sample["left_keys"], semantic_memory,
            history_weight, history_size, model.topology.feature_mode,
        )
        right_temporal = temporal_semantics(
            right_output, sample["right_keys"], semantic_memory,
            history_weight, history_size, model.topology.feature_mode,
        )
        temporal_topology = 0.5 * (left_temporal @ right_temporal.T + 1.0)
        forward = model.alpha * geometry + (1.0 - model.alpha) * temporal_topology
        reverse = model.alpha * reverse_geometry + (1.0 - model.alpha) * temporal_topology.T
        appearance = 0.5 * (
            F.normalize(left[0], dim=1) @ F.normalize(right[0], dim=1).T + 1.0
        )
        frame = int(sample["frame"])
        left_ages = update_track_ages(sample["left_keys"], frame, age_state)
        right_ages = update_track_ages(sample["right_keys"], frame, age_state)
        arrays = {
            "forward": forward.cpu().numpy(),
            "reverse": reverse.cpu().numpy(),
            "geometry": geometry.cpu().numpy(),
            "reverse_geometry": reverse_geometry.cpu().numpy(),
            "temporal": temporal_topology.cpu().numpy(),
            "current": current_topology.cpu().numpy(),
            "appearance": appearance.cpu().numpy(),
            "mask": sample["candidate_mask"].numpy(),
            "reverse_mask": sample["reverse_candidate_mask"].numpy(),
        }
        candidates = []
        features = []
        policies = []
        generator = (either_direction_topk_candidates
                     if candidate_mode == "either" else bidirectional_topk_candidates)
        for candidate in generator(
            arrays["forward"], arrays["reverse"], arrays["mask"],
            arrays["reverse_mask"], topk=candidate_topk,
        ):
            row, column = candidate.row, candidate.column
            if int(sample["left_keys"][row][3]) == int(sample["right_keys"][column][3]):
                continue
            proposed += 1
            pair_key = (
                str(sample["left_keys"][row][0]),
                int(sample["left_keys"][row][3]),
                int(sample["right_keys"][column][3]),
            )
            fused_mean = 0.5 * (
                arrays["forward"][row, column]
                + arrays["reverse"][column, row]
            )
            geometry_mean = 0.5 * (
                arrays["geometry"][row, column]
                + arrays["reverse_geometry"][column, row]
            )
            agreement = candidate_branch_agreement(
                arrays["appearance"], arrays["geometry"],
                arrays["reverse_geometry"], arrays["temporal"],
                arrays["mask"], arrays["reverse_mask"], row, column,
            )
            evidence = update_pair_multiframe_evidence(
                pair_evidence_state, pair_key, frame, float(fused_mean),
                float(geometry_mean), float(arrays["temporal"][row, column]),
                float(arrays["appearance"][row, column]), agreement,
            )
            feature = candidate_feature(
                candidate, arrays["forward"], arrays["reverse"],
                arrays["geometry"], arrays["reverse_geometry"],
                arrays["temporal"], arrays["current"], arrays["appearance"],
                arrays["mask"], arrays["reverse_mask"],
                sample["left_keys"], sample["right_keys"],
                sample["left_new_target"].numpy(),
                sample["right_new_target"].numpy(),
                sample["left"][1].numpy(), sample["right"][1].numpy(),
                sample["left"][2].numpy(), sample["right"][2].numpy(),
                left_ages, right_ages, *evidence,
            )
            features.append(torch.from_numpy(feature))
            candidates.append(candidate)
            policies.append(
                int(evidence[0]) >= minimum_support
                and float(fused_mean) >= minimum_score
                and candidate.forward_margin >= minimum_margin
                and candidate.reverse_margin >= minimum_margin
            )
        accepted = []
        if features:
            values = torch.stack(features).to(device)
            probabilities = torch.sigmoid(
                gate((values - feature_mean) / feature_scale)
            ).cpu().tolist()
            for candidate, probability, policy in zip(candidates, probabilities, policies):
                if float(probability) < gate_threshold:
                    continue
                gate_passed += 1
                if not policy:
                    continue
                policy_passed += 1
                accepted.append((
                    int(candidate.row), int(candidate.column), float(probability)
                ))
        remember_semantics(left_output, sample["left_keys"], semantic_memory, history_size)
        remember_semantics(right_output, sample["right_keys"], semantic_memory, history_size)
        records.append({
            "scene": str(sample["scene_id"]),
            "frame": frame,
            "left_keys": sample["left_keys"],
            "right_keys": sample["right_keys"],
            "pairs": accepted,
            "truth": {(int(i), int(j)) for i, j in sample["positive_pairs"].tolist()},
            "bucket": "gated",
        })
    return records, {
        "candidate_topk": int(candidate_topk),
        "candidate_mode": candidate_mode,
        "gate_proposed_rows": proposed,
        "gate_probability_passed_rows": gate_passed,
        "gate_policy_passed_rows": policy_passed,
    }


def apply_synchronous_online(records, dataset, mia_results: Path, out: Path,
                             conflict_policy: str = "support",
                             minimum_pair_support: int = 1,
                             maximum_pair_gap: int = 1,
                             minimum_pair_ema: float = 0.0,
                             evidence_decay: float = 0.7,
                             causal_collision_guard: bool = False,
                             maximum_correction_age: int = 0,
                             retroactive_confirmation: bool = False) -> dict:
    """Causally assign identities from current-frame accepted pairs only."""
    scene_payloads = {}
    scene_outputs = {}
    local_to_global = {}
    born = {}
    support = defaultdict(int)
    next_identity = defaultdict(int)
    link_counts = defaultdict(int)
    correction_counts = defaultdict(int)
    collision_rejections = defaultdict(int)
    pending_rejections = defaultdict(int)
    evidence_rejections = defaultdict(int)
    pair_evidence = {}
    token_frames = defaultdict(set)
    observed_token_frames = defaultdict(set)
    emitted_positions = defaultdict(list)
    identity_owners = defaultdict(set)
    age_rejections = defaultdict(int)

    def allocate(scene, token, frame):
        identity = next_identity[scene]
        next_identity[scene] += 1
        local_to_global[token] = identity
        born[(scene, identity)] = frame
        return identity

    for record, sample in zip(records, dataset.samples):
        scene, frame = str(record["scene"]), int(record["frame"])
        if scene not in scene_payloads:
            payloads = [load_scene_rows(mia_results, scene, 1), load_scene_rows(mia_results, scene, 2)]
            scene_payloads[scene] = payloads
            scene_outputs[scene] = [{key: [list(row) for row in rows] for key, rows in payload.items()}
                                    for payload in payloads]
            # Register the complete frozen within-view tracks before applying
            # any correction.  A residual cross-view merge is unsafe if it
            # would give two distinct source tracks the same identity in an
            # overlapping frame, including a collision that occurs only in a
            # future frame.  Looking at the full baseline track support is
            # label-free and does not introduce future appearance evidence.
            if not causal_collision_guard:
                for view, payload in enumerate(payloads, 1):
                    for key, rows in payload.items():
                        payload_frame = int(key.split("=", 1)[1])
                        for token in row_tokens(scene, view, rows):
                            baseline_id = int(token[2])
                            token_frames[token].add(payload_frame)
                            local_to_global.setdefault(token, baseline_id)
                            identity_owners[(scene, view, baseline_id)].add(token)
                            born[(scene, baseline_id)] = min(
                                born.get((scene, baseline_id), payload_frame), payload_frame
                            )
                            next_identity[scene] = max(next_identity[scene], baseline_id + 1)
        payloads = scene_payloads[scene]
        frame_rows = [payloads[view].get(f"frame={frame}", []) for view in range(2)]
        tokens = [row_tokens(scene, view + 1, frame_rows[view]) for view in range(2)]
        expected = [
            [(key[0], int(key[1]), int(key[3]), int(key[4])) for key in sample[side + "_keys"]]
            for side in ("left", "right")
        ]
        if tokens != expected:
            raise ValueError(f"frame cache and MIA row order differ at scene={scene} frame={frame}")

        # Residual identity initialization: the frozen official MIA result is
        # the no-op lower bound.  A topology branch with no accepted evidence
        # therefore reproduces every baseline ID exactly instead of rebuilding
        # all identities from scratch.
        for view_tokens in tokens:
            for token in view_tokens:
                observed_token_frames[token].add(frame)
                if token not in local_to_global:
                    baseline_id = int(token[2])
                    next_identity[scene] = max(next_identity[scene], baseline_id + 1)
                    if causal_collision_guard:
                        owners = identity_owners[(scene, int(token[1]), baseline_id)]
                        occupied_now = any(
                            other != token and frame in token_frames[other]
                            for other in owners
                        )
                        if occupied_now:
                            baseline_id = allocate(scene, token, frame)
                        else:
                            local_to_global[token] = baseline_id
                    else:
                        local_to_global[token] = baseline_id
                    identity_owners[(scene, int(token[1]), baseline_id)].add(token)
                    born.setdefault((scene, baseline_id), frame)
                if causal_collision_guard:
                    token_frames[token].add(frame)
                    identity_owners[
                        (scene, int(token[1]), int(local_to_global[token]))
                    ].add(token)

        def reassign_without_collision(token, target_id):
            old_id = local_to_global[token]
            if old_id == target_id:
                return True
            if (maximum_correction_age > 0
                    and len(observed_token_frames[token]) > maximum_correction_age):
                return False
            view = int(token[1])
            frames = token_frames[token]
            owners = identity_owners[(scene, view, target_id)]
            if any(other != token and frames.intersection(token_frames[other]) for other in owners):
                return False
            identity_owners[(scene, view, old_id)].discard(token)
            owners.add(token)
            local_to_global[token] = target_id
            if retroactive_confirmation:
                for output_view, output_frame, output_index in emitted_positions[token]:
                    scene_outputs[scene][output_view][output_frame][output_index][0] = target_id
            return True

        for left_index, right_index, _score in record["pairs"]:
            left_token, right_token = tokens[0][left_index], tokens[1][right_index]
            left_id = local_to_global.get(left_token)
            right_id = local_to_global.get(right_token)
            evidence_key = (left_token, right_token)
            previous = pair_evidence.get(evidence_key)
            if (
                previous is not None
                and 0 < frame - previous[0] <= maximum_pair_gap
            ):
                consecutive_support = previous[1] + 1
                score_ema = (
                    evidence_decay * previous[2]
                    + (1.0 - evidence_decay) * float(_score)
                )
            else:
                consecutive_support = 1
                score_ema = float(_score)
            pair_evidence[evidence_key] = (frame, consecutive_support, score_ema)
            if left_id != right_id and consecutive_support < minimum_pair_support:
                pending_rejections[scene] += 1
                continue
            if left_id != right_id and score_ema < minimum_pair_ema:
                evidence_rejections[scene] += 1
                continue
            if left_id is None and right_id is None:
                shared = allocate(scene, left_token, frame)
                local_to_global[right_token] = shared
            elif left_id is None:
                local_to_global[left_token] = right_id
            elif right_id is None:
                local_to_global[right_token] = left_id
            elif left_id != right_id:
                # The identity established earlier wins.  Standard online
                # mode changes only future rows; fixed-delay confirmation may
                # also rewrite the explicitly buffered young-track rows.
                if conflict_policy == "support":
                    left_rank = (-support[(scene, left_id)], born[(scene, left_id)], left_id)
                    right_rank = (-support[(scene, right_id)], born[(scene, right_id)], right_id)
                else:
                    left_rank = (born[(scene, left_id)], left_id)
                    right_rank = (born[(scene, right_id)], right_id)
                if left_rank <= right_rank:
                    choices = ((right_token, left_id), (left_token, right_id))
                else:
                    choices = ((left_token, right_id), (right_token, left_id))
                corrected = False
                for token, target_id in choices:
                    if reassign_without_collision(token, target_id):
                        corrected = True
                        correction_counts[scene] += 1
                        break
                if not corrected:
                    if (maximum_correction_age > 0
                            and all(len(observed_token_frames[token]) > maximum_correction_age
                                    for token, _target_id in choices)):
                        age_rejections[scene] += 1
                    else:
                        collision_rejections[scene] += 1
            link_counts[scene] += 1

        for view in range(2):
            output_rows = scene_outputs[scene][view].get(f"frame={frame}", [])
            for index, token in enumerate(tokens[view]):
                identity = local_to_global.get(token)
                if identity is None:
                    identity = allocate(scene, token, frame)
                output_rows[index] = [identity, *frame_rows[view][index][1:]]
                emitted_positions[token].append((view, f"frame={frame}", index))
                support[(scene, identity)] += 1

    out.mkdir(parents=True, exist_ok=True)
    scenes = sorted(scene_outputs, key=int)
    for scene in scenes:
        for view in range(2):
            (out / f"{scene}-{view + 1}.json").write_text(
                json.dumps(scene_outputs[scene][view]), encoding="utf-8"
            )
    return {
        "scenes": len(scenes),
        "accepted_current_frame_links": int(sum(link_counts.values())),
        "topology_identity_corrections": int(sum(correction_counts.values())),
        "collision_rejected_corrections": int(sum(collision_rejections.values())),
        "age_rejected_corrections": int(sum(age_rejections.values())),
        "support_pending_links": int(sum(pending_rejections.values())),
        "ema_rejected_links": int(sum(evidence_rejections.values())),
        "assigned_global_ids": int(sum(next_identity.values())),
        "conflict_policy": conflict_policy,
        "minimum_pair_support": minimum_pair_support,
        "maximum_pair_gap": maximum_pair_gap,
        "minimum_pair_ema": minimum_pair_ema,
        "evidence_decay": evidence_decay,
        "causal_collision_guard": causal_collision_guard,
        "maximum_correction_age": maximum_correction_age,
        "retroactive_confirmation": retroactive_confirmation,
        "fallback": "official MIA IDs are unchanged when topology accepts no corrective link",
        "inference": (
            "strictly chronological with fixed young-track confirmation buffer"
            if retroactive_confirmation else
            "strictly chronological; no future scores, median aggregation, or retroactive relabeling"
        ),
    }


def apply_synchronous_global_remap(records, mia_results: Path, out: Path,
                                   minimum_pair_support: int = 2,
                                   minimum_score: float = 0.0,
                                   maximum_track_overlap: int = 0,
                                   minimum_overlap_iou: float = 0.5) -> dict:
    """Offline whole-track remap from repeated topology evidence.

    Official MIA links are the starting components.  A topology edge is added
    only when it has repeated frame support and cannot make two source tracks
    from the same view overlap.  Accepted components are remapped consistently
    over the complete sequence, avoiding a mid-track identity discontinuity.
    """
    import numpy as np
    from scipy.optimize import linear_sum_assignment

    evidence = defaultdict(list)
    scenes = set()
    for record in records:
        scene = str(record["scene"]); scenes.add(scene)
        frame = int(record["frame"])
        for row, column, score in record["pairs"]:
            if float(score) < minimum_score:
                continue
            left = (scene, 1, int(record["left_keys"][row][3]))
            right = (scene, 2, int(record["right_keys"][column][3]))
            if left[2] != right[2]:
                evidence[(left, right)].append((frame, float(score)))

    scene_candidates = defaultdict(list)
    for scene in scenes:
        pairs = [(key, values) for key, values in evidence.items() if key[0][0] == scene
                 and len({frame for frame, _ in values}) >= minimum_pair_support]
        left = sorted({key[0] for key, _ in pairs})
        right = sorted({key[1] for key, _ in pairs})
        if not left or not right:
            continue
        scores = np.full((len(left), len(right)), -1.0, dtype=np.float32)
        supports = np.zeros_like(scores, dtype=np.int32)
        left_index = {value: index for index, value in enumerate(left)}
        right_index = {value: index for index, value in enumerate(right)}
        for (lkey, rkey), values in pairs:
            i, j = left_index[lkey], right_index[rkey]
            scores[i, j] = float(np.median([score for _, score in values]))
            supports[i, j] = len({frame for frame, _ in values})
        rows, columns = linear_sum_assignment(-scores)
        for i, j in zip(rows, columns):
            if scores[i, j] >= 0 and supports[i, j] >= minimum_pair_support:
                scene_candidates[scene].append(
                    (left[i], right[j], float(scores[i, j]), int(supports[i, j]))
                )

    out.mkdir(parents=True, exist_ok=True)
    accepted_total = rejected_total = 0
    scene_report = {}
    for scene in sorted(scenes, key=int):
        payloads = [load_scene_rows(mia_results, scene, view) for view in (1, 2)]
        nodes = []
        frames = defaultdict(set)
        boxes = defaultdict(lambda: defaultdict(list))
        for view, payload in enumerate(payloads, 1):
            for frame_key, rows in payload.items():
                frame = int(frame_key.split("=", 1)[1])
                for row in rows:
                    node = (scene, view, int(row[0]))
                    frames[node].add(frame)
                    boxes[node][frame].append(tuple(float(value) for value in row[1:5]))
            nodes.extend((scene, view, identity) for identity in ids_in_payload(payload))

        union = UnionFind()
        members = {}
        for node in nodes:
            union.find(node); members[node] = {node}

        def merge(first, second):
            root_first, root_second = union.find(first), union.find(second)
            if root_first == root_second:
                return True, 0
            combined = members[root_first] | members[root_second]
            overlap_frames = set()
            overlap_pairs = []
            for view in (1, 2):
                view_nodes = [node for node in combined if node[1] == view]
                for index, node in enumerate(view_nodes):
                    for other in view_nodes[index + 1:]:
                        common_frames = frames[node].intersection(frames[other])
                        overlap_frames.update(common_frames)
                        overlap_pairs.extend((node, other, frame) for frame in common_frames)
            overlap_count = len(overlap_frames)
            if overlap_count:
                if maximum_track_overlap <= 0 or overlap_count > maximum_track_overlap:
                    return False, overlap_count
                for node, other, frame in overlap_pairs:
                    best_iou = 0.0
                    for first_box in boxes[node][frame]:
                        for second_box in boxes[other][frame]:
                            ix1 = max(first_box[0], second_box[0]); iy1 = max(first_box[1], second_box[1])
                            ix2 = min(first_box[2], second_box[2]); iy2 = min(first_box[3], second_box[3])
                            intersection = max(ix2 - ix1, 0.0) * max(iy2 - iy1, 0.0)
                            first_area = max(first_box[2] - first_box[0], 0.0) * max(first_box[3] - first_box[1], 0.0)
                            second_area = max(second_box[2] - second_box[0], 0.0) * max(second_box[3] - second_box[1], 0.0)
                            union_area = first_area + second_area - intersection
                            best_iou = max(best_iou, intersection / max(union_area, 1e-12))
                    if best_iou < minimum_overlap_iou:
                        return False, overlap_count
            union.union(root_first, root_second)
            new_root = union.find(root_first)
            members[new_root] = combined
            if root_first != new_root:
                members.pop(root_first, None)
            if root_second != new_root:
                members.pop(root_second, None)
            return True, 0

        # Preserve every official cross-view MIA identity before proposing
        # residual topology links.
        common = ids_in_payload(payloads[0]) & ids_in_payload(payloads[1])
        for identity in common:
            merge((scene, 1, identity), (scene, 2, identity))

        accepted = rejected = 0
        rejected_overlaps = []
        candidates = sorted(
            scene_candidates.get(scene, []), key=lambda row: (-row[3], -row[2])
        )
        for left, right, _score, _support in candidates:
            merged, overlap = merge(left, right)
            if merged:
                accepted += 1
            else:
                rejected += 1
                rejected_overlaps.append(int(overlap))
        accepted_total += accepted; rejected_total += rejected

        roots = sorted({union.find(node) for node in nodes}, key=lambda value: (value[1], value[2]))
        component_id = {root: index for index, root in enumerate(roots)}
        suppressed_duplicates = 0
        for view, payload in enumerate(payloads, 1):
            remapped = {}
            for frame, rows in payload.items():
                grouped = defaultdict(list)
                for row in rows:
                    original_id = int(row[0])
                    mapped_id = component_id[union.find((scene, view, original_id))]
                    grouped[mapped_id].append((original_id, [mapped_id, *row[1:]]))
                output_rows = []
                for mapped_id, group in grouped.items():
                    if len({original_id for original_id, _ in group}) <= 1:
                        output_rows.extend(row for _, row in group)
                        continue
                    selected = max(
                        (row for _, row in group),
                        key=lambda row: max(float(row[3]) - float(row[1]), 0.0)
                                        * max(float(row[4]) - float(row[2]), 0.0),
                    )
                    output_rows.append(selected)
                    suppressed_duplicates += len(group) - 1
                remapped[frame] = output_rows
            (out / f"{scene}-{view}.json").write_text(json.dumps(remapped), encoding="utf-8")
        scene_report[scene] = {
            "candidate_links": len(candidates), "accepted_links": accepted,
            "collision_rejected_links": rejected, "components": len(roots),
            "suppressed_overlap_duplicates": suppressed_duplicates,
            "rejected_overlap_frames": {
                "min": min(rejected_overlaps) if rejected_overlaps else 0,
                "median": float(np.median(rejected_overlaps)) if rejected_overlaps else 0.0,
                "max": max(rejected_overlaps) if rejected_overlaps else 0,
                "at_most_1": sum(value <= 1 for value in rejected_overlaps),
                "at_most_2": sum(value <= 2 for value in rejected_overlaps),
                "at_most_5": sum(value <= 5 for value in rejected_overlaps),
            },
        }
    return {
        "mode": "offline_topology_whole_track_remap",
        "minimum_pair_support": minimum_pair_support,
        "minimum_score": minimum_score,
        "maximum_track_overlap": maximum_track_overlap,
        "minimum_overlap_iou": minimum_overlap_iou,
        "accepted_topology_links": accepted_total,
        "collision_rejected_links": rejected_total,
        "scenes": scene_report,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--prediction", default="")
    parser.add_argument("--geometry-cache", default="")
    parser.add_argument("--mia-results", required=True, type=Path)
    parser.add_argument("--threshold", type=float, default=0.0,
                        help="Legacy tracklet ablation only; paper frame inference uses Eq. (15) alone")
    parser.add_argument("--alpha", type=float,
                        help="Validation-selected fusion weight; defaults to checkpoint value")
    parser.add_argument("--history-weight", type=float)
    parser.add_argument("--history-size", type=int)
    parser.add_argument("--base-margin", type=float, required=True)
    parser.add_argument("--candidate-growth", type=float, required=True)
    parser.add_argument("--minimum-score", type=float, default=0.0,
                        help="Validation-frozen minimum bidirectional fused score")
    parser.add_argument("--conflict-policy", choices=("support", "earliest"), default="support",
                        help="Validation-selectable causal ID conflict resolution")
    parser.add_argument("--minimum-pair-support", type=int, default=1)
    parser.add_argument("--maximum-pair-gap", type=int, default=1)
    parser.add_argument("--minimum-pair-ema", type=float, default=0.0)
    parser.add_argument("--evidence-decay", type=float, default=0.7)
    parser.add_argument("--causal-collision-guard", action="store_true")
    parser.add_argument("--maximum-correction-age", type=int, default=0,
                        help="Only reassign a source track within this many observed frames; 0 disables")
    parser.add_argument("--retroactive-confirmation", action="store_true",
                        help="Rewrite buffered rows of an accepted young track (fixed-delay inference)")
    parser.add_argument("--global-track-remap", action="store_true",
                        help="Offline whole-track topology remap with full-track collision checks")
    parser.add_argument("--candidate-topk", type=int, default=1,
                        help="Bidirectional topology candidate rank for global remap; validation-selected")
    parser.add_argument("--maximum-track-overlap", type=int, default=0,
                        help="Allow this many duplicate transition frames in offline global remap")
    parser.add_argument("--minimum-overlap-iou", type=float, default=0.5)
    parser.add_argument("--repair-gate", default="")
    parser.add_argument("--repair-gate-threshold", type=float, default=0.0)
    parser.add_argument("--gate-minimum-score", type=float, default=0.0)
    parser.add_argument("--gate-minimum-margin", type=float, default=0.0)
    parser.add_argument("--gate-minimum-support", type=int, default=1)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--timing-report", type=Path)
    args = parser.parse_args()

    total_started = time.perf_counter()
    model_load_started = time.perf_counter()
    model, config = load_model(args.checkpoint, args.device)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    model_load_seconds = time.perf_counter() - model_load_started
    model_parameters = sum(parameter.numel() for parameter in model.parameters())
    if args.alpha is not None:
        model.alpha = float(args.alpha)
    data_prepare_started = time.perf_counter()
    cache_payload = torch.load(args.cache, map_location="cpu", weights_only=False)
    synchronous = cache_payload.get("metadata", {}).get("representation") == "mia_net_synchronous_frame_targets"
    if synchronous:
        dataset = SynchronousMIAFrameDataset(payload=cache_payload)
        geometry_payload = None
        geometry = None
    else:
        if not args.prediction or not args.geometry_cache:
            parser.error("legacy tracklet ablation requires --prediction and --geometry-cache")
        dataset = PredictedTopologyFeatureDataset(args.cache, args.prediction)
        geometry_payload = torch.load(args.geometry_cache, map_location="cpu", weights_only=False)
        geometry = geometry_payload["windows"]
    validate_paper_protocol(dataset, geometry_payload, config)
    data_prepare_seconds = time.perf_counter() - data_prepare_started
    history_weight = (args.history_weight if args.history_weight is not None
                      else float(config.get("history_weight", 0.5)))
    history_size = (args.history_size if args.history_size is not None
                    else int(config.get("history_size", 4)))
    gate_report = {}
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    topology_started = time.perf_counter()
    if synchronous and args.repair_gate:
        gate_bundle = load_repair_gate(args.repair_gate, args.device)
        gate_payload = gate_bundle[3]
        candidate_topk = int(
            gate_payload.get("gate_config", {}).get("candidate_topk", 1)
        )
        candidate_mode = str(
            gate_payload.get("gate_config", {}).get("candidate_mode", "bidirectional")
        )
        if str(Path(gate_payload["topology_checkpoint"]).resolve()) != str(
            Path(args.checkpoint).resolve()
        ):
            raise ValueError("repair gate and topology checkpoint differ")
        records, gate_report = collect_gated_records(
            model, dataset, gate_bundle, args.device,
            history_weight, history_size,
            candidate_topk,
            candidate_mode,
            args.repair_gate_threshold,
            args.gate_minimum_score,
            args.gate_minimum_margin,
            args.gate_minimum_support,
        )
    else:
        records = collect(
            model, dataset, args.device, geometry,
            history_weight=history_weight, history_size=history_size,
            base_margin=args.base_margin, candidate_growth=args.candidate_growth,
            minimum_score=args.minimum_score,
            oracle_bidirectional_topk=(args.candidate_topk if args.candidate_topk > 1 else 0),
        )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    topology_seconds = time.perf_counter() - topology_started
    if synchronous:
        application_started = time.perf_counter()
        if args.global_track_remap:
            online_report = apply_synchronous_global_remap(
                records, args.mia_results, args.out,
                minimum_pair_support=args.minimum_pair_support,
                minimum_score=args.minimum_score,
                maximum_track_overlap=args.maximum_track_overlap,
                minimum_overlap_iou=args.minimum_overlap_iou,
            )
        else:
            online_report = apply_synchronous_online(
                records, dataset, args.mia_results, args.out, args.conflict_policy,
                minimum_pair_support=args.minimum_pair_support,
                maximum_pair_gap=args.maximum_pair_gap,
                minimum_pair_ema=args.minimum_pair_ema,
                evidence_decay=args.evidence_decay,
                causal_collision_guard=args.causal_collision_guard,
                maximum_correction_age=args.maximum_correction_age,
                retroactive_confirmation=args.retroactive_confirmation,
            )
        application_seconds = time.perf_counter() - application_started
        report = {
            "checkpoint": str(Path(args.checkpoint).resolve()),
            "cache": str(Path(args.cache).resolve()),
            "alpha": float(model.alpha),
            "history_weight": history_weight,
            "history_size": history_size,
            "base_margin": args.base_margin,
            "candidate_growth": args.candidate_growth,
            "minimum_score": args.minimum_score,
            "candidate_topk": args.candidate_topk,
            "repair_gate": str(Path(args.repair_gate).resolve()) if args.repair_gate else "",
            "repair_gate_threshold": args.repair_gate_threshold,
            "gate_minimum_score": args.gate_minimum_score,
            "gate_minimum_margin": args.gate_minimum_margin,
            "gate_minimum_support": args.gate_minimum_support,
            **gate_report,
            **online_report,
        }
        (args.out / "topology_application_report.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8"
        )
        if args.timing_report:
            timing_report = {
                "scope": "P1 identity-topology incremental module",
                "device": args.device,
                "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
                "dataset_items": len(dataset),
                "records": len(records),
                "parameters": int(model_parameters),
                "model_load_seconds": model_load_seconds,
                "data_prepare_seconds": data_prepare_seconds,
                "topology_inference_seconds": topology_seconds,
                "assignment_application_seconds": application_seconds,
                "incremental_processing_seconds": topology_seconds + application_seconds,
                "end_to_end_seconds": time.perf_counter() - total_started,
                "milliseconds_per_dataset_item": (
                    (topology_seconds + application_seconds) * 1000.0 / len(dataset)
                ),
                "peak_cuda_allocated_bytes": (
                    int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else None
                ),
                "peak_cuda_reserved_bytes": (
                    int(torch.cuda.max_memory_reserved()) if torch.cuda.is_available() else None
                ),
            }
            args.timing_report.parent.mkdir(parents=True, exist_ok=True)
            args.timing_report.write_text(
                json.dumps(timing_report, indent=2), encoding="utf-8"
            )
        print(json.dumps(report, indent=2))
        return
    selected = persistent_cross_view_matches(records, args.threshold)

    modeled = set()
    for record in records:
        modeled.update(track_token(key) for key in record["left_keys"])
        modeled.update(track_token(key) for key in record["right_keys"])
    selected_nodes = {node for left, right, _ in selected for node in (left, right)}
    selected_by_scene = defaultdict(list)
    for left, right, score in selected:
        selected_by_scene[left[0]].append((left, right, score, "topology"))

    prediction = torch.load(args.prediction, map_location="cpu", weights_only=False)
    scenes = sorted({str(item.scene_id) for item in prediction["items"]}, key=int)
    args.out.mkdir(parents=True, exist_ok=True)
    report = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "prediction": str(Path(args.prediction).resolve()),
        "geometry_cache": str(Path(args.geometry_cache).resolve()),
        "threshold": args.threshold,
        "alpha": float(model.alpha),
        "history_weight": history_weight,
        "history_size": history_size,
        "base_margin": args.base_margin,
        "candidate_growth": args.candidate_growth,
        "scenes": {},
    }

    for scene in scenes:
        payload1 = load_scene_rows(args.mia_results, scene, 1)
        payload2 = load_scene_rows(args.mia_results, scene, 2)
        ids1, ids2 = ids_in_payload(payload1), ids_in_payload(payload2)
        links = list(selected_by_scene.get(scene, []))

        # Preserve official MIA associations only when topology had no pair of
        # sufficiently observed nodes on both sides.  A modeled-but-rejected
        # pair remains rejected, as required by the final assignment rule.
        retained_sparse = 0
        for identity in sorted(ids1 & ids2):
            left = (scene, 1, identity)
            right = (scene, 2, identity)
            if left in selected_nodes or right in selected_nodes:
                continue
            if left not in modeled or right not in modeled:
                links.append((left, right, 1.0, "mia_sparse"))
                retained_sparse += 1

        union = UnionFind()
        for identity in ids1:
            union.find((scene, 1, identity))
        for identity in ids2:
            union.find((scene, 2, identity))
        for left, right, _, _ in links:
            union.union(left, right)

        roots = sorted({union.find(node) for node in union.parent}, key=lambda value: (value[1], value[2]))
        component_id = {root: index for index, root in enumerate(roots)}

        def remap(payload, view):
            output = {}
            for frame, rows in payload.items():
                output[frame] = [
                    [component_id[union.find((scene, view, int(row[0])))], *row[1:5]]
                    for row in rows
                ]
            return output

        (args.out / f"{scene}-1.json").write_text(json.dumps(remap(payload1, 1)), encoding="utf-8")
        (args.out / f"{scene}-2.json").write_text(json.dumps(remap(payload2, 2)), encoding="utf-8")
        report["scenes"][scene] = {
            "topology_links": sum(source == "topology" for *_, source in links),
            "retained_sparse_mia_links": retained_sparse,
            "components": len(roots),
            "view1_ids": len(ids1),
            "view2_ids": len(ids2),
        }

    (args.out / "topology_application_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps({
        "scenes": len(scenes),
        "topology_links": sum(row["topology_links"] for row in report["scenes"].values()),
        "retained_sparse_mia_links": sum(row["retained_sparse_mia_links"] for row in report["scenes"].values()),
        "alpha": report["alpha"],
        "threshold": report["threshold"],
    }, indent=2))


if __name__ == "__main__":
    main()
