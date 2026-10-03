from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from uav_tracking.identity_topology_data import (
    TopologyFeatureDataset, PredictedTopologyFeatureDataset, SynchronousMIAFrameDataset,
)
from uav_tracking.identity_topology_model import TopologyEnhancedAssociation


def validate_paper_protocol(dataset, geometry_payload, config):
    expected_frontend = config.get("expected_frontend", "autoassign_bytetrack")
    metadata = dataset.appearance_metadata
    if metadata.get("appearance_source") != "frozen_official_mia_detector_fpn":
        raise ValueError("paper evaluation requires frozen official MIA detector FPN appearance")
    if metadata.get("frontend") != expected_frontend:
        raise ValueError("appearance frontend and trained paper frontend differ")
    synchronous = isinstance(dataset, SynchronousMIAFrameDataset)
    expected_time = "current_frame_only" if synchronous else "latest_current_observation_only"
    if metadata.get("temporal_aggregation") != expected_time:
        raise ValueError("paper evaluation requires current-time rather than averaged appearance")
    prediction = getattr(dataset, "prediction_metadata", {})
    if prediction.get("report", {}).get("frontend") != expected_frontend:
        raise ValueError("paper evaluation requires matching official MIA predicted tracks")
    if synchronous:
        expected_insertion = config.get("expected_insertion_point", "post_mia_output")
        expected_pooling = config.get(
            "expected_pooling", "per_level_roi_align_1x1_then_equal_fpn_mean"
        )
        if metadata.get("insertion_point", "post_mia_output") != expected_insertion:
            raise ValueError("topology insertion point and trained checkpoint differ")
        if metadata.get("pooling") != expected_pooling:
            raise ValueError("frozen FPN pooling and trained checkpoint differ")
        if metadata.get("topology_time") != "all nodes are targets observed in the same synchronized frame t":
            raise ValueError("topology cache mixes nodes from different times")
        if (metadata.get("geometry_direction") != "ordered bidirectional view1_to_view2 and view2_to_view1"
                or metadata.get("candidate_rule") != "all synchronized target pairs; geometry is soft and never deletes candidates"
                or metadata.get("geometry_reachability") != "diagnostic only; never used as a candidate mask"
                or metadata.get("geometry_scales") != {"new": 80.0, "existing": 50.0}
                or metadata.get("geometry_state") != "official MIA new iff current ID exceeds previous committed maximum ID"):
            raise ValueError("embedded geometry does not satisfy the paper protocol")
        return
    protocol = (geometry_payload or {}).get("protocol", {})
    if (protocol.get("direction") != "view1_to_view2"
            or protocol.get("candidate_rule") != "temporal co-visibility; no geometric-score threshold"
            or float(protocol.get("new_target_scale", -1)) != 80.0
            or float(protocol.get("existing_target_scale", -1)) != 50.0):
        raise ValueError("geometry cache does not satisfy the paper's directed soft MIA protocol")


def load_model(checkpoint: str | Path, device: str):
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    cfg = state["config"]
    model = TopologyEnhancedAssociation(
        state["appearance_dim"], alpha=cfg.get("alpha", 0.5),
        context_weight=cfg.get("score_context_weight", cfg.get("context_weight", 0.5)),
        hidden_dim=cfg["hidden_dim"], patterns=cfg["patterns"],
        propagation_layers=cfg.get("propagation_layers", 1),
        purification=cfg["purification"], topology_mode=cfg["topology_mode"],
        pattern_momentum=cfg.get("pattern_momentum", 1.0),
        feature_mode=cfg.get("feature_mode", "h_plus_g"),
        purification_floor=cfg.get("purification_floor", 0.05),
        temperature=cfg.get("relation_temperature", 0.10),
    ).to(device)
    model.load_state_dict(state["model"])
    model.eval()
    return model, cfg


def _bucket(sample) -> str:
    left_app, _, _ = sample["left"]
    right_app, _, _ = sample["right"]
    pairs = sample["positive_pairs"]
    quality = float(sample.get("mean_quality", 1.0))
    if float(sample.get("mean_occlusion", 0.0)) >= 0.15 or quality < 0.80:
        return "Occlusion"
    if len(left_app) + len(right_app) >= 24:
        return "Dense Targets"
    similarity = left_app @ right_app.T
    positives = {(int(i), int(j)) for i, j in pairs.tolist()}
    negatives = [float(similarity[i, j]) for i in range(len(left_app)) for j in range(len(right_app))
                 if (i, j) not in positives]
    if negatives and max(negatives) >= 0.82:
        return "Similar Appearance"
    positive_scores = [float(similarity[i, j]) for i, j in positives]
    if positive_scores and float(np.mean(positive_scores)) < 0.45:
        return "Viewpoint Variation"
    return "Normal"


def _empty_counts():
    return {"correct": 0, "false": 0, "missed": 0, "possible": 0, "ids": 0, "windows": 0}


def finalize(counts):
    c, f, m, p, ids = (counts[k] for k in ("correct", "false", "missed", "possible", "ids"))
    return {
        **counts,
        "mota": 1.0 - (f + m + ids) / max(p, 1),
        "idf1": 2.0 * c / max(2 * c + f + m, 1),
        "mda": c / max(p + f + m, 1),
    }


def competition_assignment(scores, candidate_mask=None, base_margin=0.0, candidate_growth=0.0,
                           reverse_scores=None, reverse_candidate_mask=None,
                           minimum_score=0.0):
    """Eq. (15): one-to-one matches accepted by bidirectional competition."""
    if candidate_mask is None:
        candidate_mask = np.ones_like(scores, dtype=bool)
    if reverse_scores is None:
        reverse_scores = scores.T
    if reverse_candidate_mask is None:
        reverse_candidate_mask = candidate_mask.T
    if (scores.size == 0 or reverse_scores.size == 0
            or not candidate_mask.any() or not reverse_candidate_mask.any()):
        return []
    masked = np.where(candidate_mask, scores, -np.inf)
    reverse_masked = np.where(reverse_candidate_mask, reverse_scores, -np.inf)
    row_best = masked.argmax(axis=1)
    reverse_best = reverse_masked.argmax(axis=1)
    accepted = []
    for row, col in enumerate(row_best):
        if not candidate_mask[row, col]:
            continue
        # The paper first selects the highest scoring candidate in each
        # direction.  Bidirectional identity confirmation therefore requires
        # mutual best candidates, not a scene-level Hungarian reassignment.
        if not reverse_candidate_mask[col, row] or int(reverse_best[col]) != row:
            continue
        row_candidates = np.flatnonzero(candidate_mask[row])
        reverse_candidates = np.flatnonzero(reverse_candidate_mask[col])
        row_other = scores[row, row_candidates[row_candidates != col]]
        reverse_other = reverse_scores[col, reverse_candidates[reverse_candidates != row]]
        row_margin = float(scores[row, col] - row_other.max()) if row_other.size else float("inf")
        reverse_margin = (float(reverse_scores[col, row] - reverse_other.max())
                          if reverse_other.size else float("inf"))
        # Paper Eq. (15): a0 + b0 log(N + 1), applied in both ordered
        # view directions for bidirectional competition.
        row_required = base_margin + candidate_growth * np.log(len(row_candidates) + 1.0)
        reverse_required = base_margin + candidate_growth * np.log(len(reverse_candidates) + 1.0)
        bidirectional_score = float(0.5 * (scores[row, col] + reverse_scores[col, row]))
        if (row_margin > row_required and reverse_margin > reverse_required
                and bidirectional_score >= minimum_score):
            accepted.append((
                int(row), int(col), bidirectional_score,
            ))
    return accepted


@torch.no_grad()
def collect(model, dataset, device: str, geometry_windows=None, batch_size: int = 8,
            history_weight=0.5, history_size=4, base_margin=0.0, candidate_growth=0.0,
            oracle_bidirectional_topk: int = 0, minimum_score: float = 0.0):
    records = []
    from collections import defaultdict, deque
    from train_identity_topology import geometry_for_sample, temporal_semantics, remember_semantics
    memory = defaultdict(lambda: deque(maxlen=history_size))
    for index in range(len(dataset)):
        sample = dataset[index]
        geometric, reverse_geometric, candidate_mask, reverse_candidate_mask = geometry_for_sample(
            dataset, index, sample, geometry_windows, device
        )
        left = tuple(value.to(device) for value in sample["left"])
        right = tuple(value.to(device) for value in sample["right"])
        _, left_output, right_output = model(left, right, geometric)
        left_semantic = temporal_semantics(
            left_output, sample["left_keys"], memory, history_weight, history_size,
            model.topology.feature_mode,
        )
        right_semantic = temporal_semantics(
            right_output, sample["right_keys"], memory, history_weight, history_size,
            model.topology.feature_mode,
        )
        topology = 0.5 * (left_semantic @ right_semantic.T + 1.0)
        scores = (model.alpha * geometric + (1.0 - model.alpha) * topology).cpu().numpy()
        reverse_scores = (
            model.alpha * reverse_geometric + (1.0 - model.alpha) * topology.T
        ).cpu().numpy()
        forward_mask = candidate_mask.cpu().numpy()
        reverse_mask = reverse_candidate_mask.cpu().numpy()
        if oracle_bidirectional_topk > 0:
            # Validation-only candidate-space audit.  Formal inference keeps
            # Eq. (15) mutual Top-1; this mode exposes whether a high-recall
            # Top-K generator could support a separate precision gate.
            k = int(oracle_bidirectional_topk)
            forward_rank = np.full(scores.shape, scores.shape[1] + 1, dtype=np.int32)
            reverse_rank = np.full(reverse_scores.shape, reverse_scores.shape[1] + 1, dtype=np.int32)
            for row in range(scores.shape[0]):
                columns = np.flatnonzero(forward_mask[row])
                order = columns[np.argsort(-scores[row, columns], kind="stable")]
                forward_rank[row, order] = np.arange(1, len(order) + 1)
            for row in range(reverse_scores.shape[0]):
                columns = np.flatnonzero(reverse_mask[row])
                order = columns[np.argsort(-reverse_scores[row, columns], kind="stable")]
                reverse_rank[row, order] = np.arange(1, len(order) + 1)
            selected = (
                forward_mask
                & reverse_mask.T
                & (forward_rank <= k)
                & (reverse_rank.T <= k)
            )
            pairs = [
                (int(row), int(column), float(0.5 * (
                    scores[row, column] + reverse_scores[column, row]
                )))
                for row, column in np.argwhere(selected)
            ]
        else:
            pairs = competition_assignment(
                scores, forward_mask, base_margin, candidate_growth,
                reverse_scores=reverse_scores,
                reverse_candidate_mask=reverse_mask,
                minimum_score=minimum_score,
            )
        remember_semantics(left_output, sample["left_keys"], memory, history_size)
        remember_semantics(right_output, sample["right_keys"], memory, history_size)
        window = dataset.windows[index] if hasattr(dataset, "windows") else None
        records.append({
            "scene": window.scene_id if window is not None else str(sample["scene_id"]),
            "frame": int(sample.get("frame", sample.get("start", 0))),
            "left_keys": sample["left_keys"], "right_keys": sample["right_keys"],
            "pairs": pairs,
            "truth": {(int(i), int(j)) for i, j in sample["positive_pairs"].tolist()},
            "bucket": _bucket(sample),
        })
    return records


def summarize(records, threshold: float):
    overall = _empty_counts()
    scenarios = defaultdict(_empty_counts)
    last_assignment = {}
    repair_correct = repair_false = repair_missed = repair_possible = 0
    for record in records:
        truth = record["truth"]
        selected = {(i, j) for i, j, score in record["pairs"] if score >= threshold}
        correct = len(selected & truth)
        false = len(selected - truth)
        missed = len(truth - selected)
        repair_selected = {
            (i, j) for i, j in selected
            if int(record["left_keys"][i][3]) != int(record["right_keys"][j][3])
        }
        repair_truth = {
            (i, j) for i, j in truth
            if int(record["left_keys"][i][3]) != int(record["right_keys"][j][3])
        }
        repair_correct += len(repair_selected & repair_truth)
        repair_false += len(repair_selected - repair_truth)
        repair_missed += len(repair_truth - repair_selected)
        repair_possible += len(repair_truth)
        ids = 0
        for row, col in selected:
            left_key = record["left_keys"][row]
            right_key = record["right_keys"][col]
            state_key = (left_key[0], left_key[3], left_key[4] if len(left_key) > 4 else 0)
            predicted_right = (right_key[3], right_key[4] if len(right_key) > 4 else 0)
            if state_key in last_assignment and last_assignment[state_key] != predicted_right:
                ids += 1
            last_assignment[state_key] = predicted_right
        bucket = record["bucket"]
        for target in (overall, scenarios[bucket]):
            target["correct"] += correct; target["false"] += false
            target["missed"] += missed; target["possible"] += len(truth)
            target["ids"] += ids; target["windows"] += 1
    repair_precision = repair_correct / max(repair_correct + repair_false, 1)
    repair_recall = repair_correct / max(repair_possible, 1)
    repair_f0_5 = (1.25 * repair_precision * repair_recall
                   / max(0.25 * repair_precision + repair_recall, 1e-12))
    overall_result = finalize(overall)
    overall_result.update({
        "repair_correct": repair_correct,
        "repair_false": repair_false,
        "repair_missed": repair_missed,
        "repair_possible": repair_possible,
        "repair_precision": repair_precision,
        "repair_recall": repair_recall,
        "repair_f0_5": repair_f0_5,
    })
    return {"overall": overall_result,
            "scenarios": {name: finalize(value) for name, value in sorted(scenarios.items())}}


def main():
    parser = argparse.ArgumentParser(description="Internal validation-only threshold selection for training")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--xml", default="data/datafull/new_xml")
    parser.add_argument("--prediction", default="",
                        help="predicted_windows.pt; enables detector/tracklet evaluation")
    parser.add_argument("--geometry-cache", default="",
                        help="Frozen MIA-Net geometric score cache")
    parser.add_argument("--threshold", type=float, default=0.0)
    parser.add_argument("--alpha", type=float,
                        help="Override checkpoint fusion weight for validation selection")
    parser.add_argument("--tune-threshold", action="store_true",
                        help="Select threshold on this cache only; use for validation, never test")
    parser.add_argument("--out", required=True)
    parser.add_argument("--history-weight", type=float, default=0.5)
    parser.add_argument("--history-size", type=int, default=4)
    parser.add_argument("--base-margin", type=float, default=0.0)
    parser.add_argument("--candidate-growth", type=float, default=0.0)
    parser.add_argument("--minimum-score", type=float, default=0.0,
                        help="Validation-selected minimum bidirectional fused score")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    payload = torch.load(args.cache, map_location="cpu", weights_only=False)
    if payload.get("metadata", {}).get("split") != "val":
        parser.error("This training helper accepts validation caches only; use test.py for final test metrics")
    model, config = load_model(args.checkpoint, args.device)
    if args.alpha is not None:
        model.alpha = float(args.alpha)
    if args.prediction:
        dataset = PredictedTopologyFeatureDataset(args.cache, args.prediction)
    else:
        if payload.get("metadata", {}).get("representation") == "mia_net_synchronous_frame_targets":
            dataset = SynchronousMIAFrameDataset(payload=payload)
        else:
            dataset = TopologyFeatureDataset(args.cache, args.xml)
    geometry_windows = None
    if args.geometry_cache:
        geometry_payload = torch.load(args.geometry_cache, map_location="cpu", weights_only=False)
        validate_paper_protocol(dataset, geometry_payload, config)
        geometry_windows = geometry_payload["windows"]
    elif isinstance(dataset, SynchronousMIAFrameDataset):
        validate_paper_protocol(dataset, None, config)
    records = collect(
        model, dataset, args.device, geometry_windows,
        history_weight=args.history_weight, history_size=args.history_size,
        base_margin=args.base_margin, candidate_growth=args.candidate_growth,
        minimum_score=args.minimum_score,
    )
    sweep = []
    if args.tune_threshold:
        for threshold in np.linspace(-0.2, 0.95, 48):
            metrics = summarize(records, float(threshold))["overall"]
            sweep.append({"threshold": float(threshold), **metrics})
        selected = max(sweep, key=lambda row: (row["mda"], row["idf1"], -row["threshold"]))
        threshold = selected["threshold"]
    else:
        threshold = args.threshold
    result = summarize(records, threshold)
    result.update({"checkpoint": str(args.checkpoint), "cache": str(args.cache),
                   "threshold": threshold, "threshold_sweep": sweep,
                   "alpha": float(model.alpha), "training_config": config})
    result.update({"history_weight": args.history_weight, "history_size": args.history_size,
                   "base_margin": args.base_margin, "candidate_growth": args.candidate_growth})
    result["minimum_score"] = args.minimum_score
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result["overall"], indent=2))


if __name__ == "__main__":
    main()
