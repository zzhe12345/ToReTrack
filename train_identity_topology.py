from __future__ import annotations

import argparse
import json
from pathlib import Path
from collections import defaultdict, deque
import random

import torch
import torch.nn.functional as F

from uav_tracking.identity_topology_data import (
    TopologyFeatureDataset, PredictedTopologyFeatureDataset, SynchronousMIAFrameDataset,
)
from uav_tracking.identity_topology_model import (
    TopologyEnhancedAssociation, TopologyOutput,
)


def collate(samples, device):
    left_lengths = [len(sample["left"][0]) for sample in samples]
    right_lengths = [len(sample["right"][0]) for sample in samples]

    def padded(side, lengths):
        result = []
        maximum = max(lengths)
        for feature in range(3):
            dim = samples[0][side][feature].shape[1]
            tensor = torch.zeros(len(samples), maximum, dim, device=device)
            for batch, sample in enumerate(samples):
                value = sample[side][feature].to(device)
                tensor[batch, :len(value)] = value
            result.append(tensor)
        mask = torch.arange(maximum, device=device)[None] < torch.tensor(lengths, device=device)[:, None]
        return tuple(result), mask

    left, left_mask = padded("left", left_lengths)
    right, right_mask = padded("right", right_lengths)
    geometric = left[0].new_zeros((len(samples), left[0].shape[1], right[0].shape[1]))
    pairs = [sample["positive_pairs"].to(device) for sample in samples]
    return left, right, left_mask, right_mask, geometric, pairs, left_lengths, right_lengths


def slice_output(output: TopologyOutput, batch: int, count: int) -> TopologyOutput:
    return TopologyOutput(*(value[batch, :count] for value in (
        output.nodes, output.memberships, output.reliability,
        output.identity, output.context, output.semantic,
    )))


def association_loss(scores: torch.Tensor, pairs: torch.Tensor, temperature: float = 0.07,
                     candidate_mask: torch.Tensor | None = None,
                     reverse_scores: torch.Tensor | None = None,
                     reverse_candidate_mask: torch.Tensor | None = None,
                     positive_weights: torch.Tensor | None = None) -> torch.Tensor:
    if pairs.numel() == 0:
        return scores.sum() * 0.0
    if candidate_mask is None:
        candidate_mask = torch.ones_like(scores, dtype=torch.bool)
    if reverse_scores is None:
        reverse_scores = scores.T
    if reverse_candidate_mask is None:
        reverse_candidate_mask = candidate_mask.T
    rows, columns = pairs[:, 0], pairs[:, 1]
    keep = candidate_mask[rows, columns] & reverse_candidate_mask[columns, rows]
    rows, columns = rows[keep], columns[keep]
    if len(rows) == 0:
        return scores.sum() * 0.0
    if positive_weights is None:
        positive_weights = scores.new_ones(len(pairs))
    weights = positive_weights.to(scores.device)[keep]
    weights = weights / weights.sum().clamp_min(1e-8)
    logits = scores.masked_fill(~candidate_mask, -1e4)
    reverse_logits = reverse_scores.masked_fill(~reverse_candidate_mask, -1e4)
    return 0.5 * (
        (F.cross_entropy(logits[rows] / temperature, columns, reduction="none") * weights).sum()
        + (F.cross_entropy(
            reverse_logits[columns] / temperature, rows, reduction="none"
        ) * weights).sum()
    )


def apply_geometry_batch(dataset, indices, samples, geometric, pairs_list, geometry_windows, device):
    """Fill frozen official MIA scores without geometry-threshold pruning."""
    candidate_masks = []
    if geometry_windows is None:
        return pairs_list, candidate_masks
    filtered_pairs = []
    for batch, (index, sample, pairs) in enumerate(zip(indices, samples, pairs_list)):
        entry = geometry_windows[(str(sample["scene_id"]), int(sample["start"]))]
        window = dataset.windows[index]
        expected_left = [dataset.keys[i] for i in window.left_indices]
        expected_right = [dataset.keys[i] for i in window.right_indices]
        if ([tuple(x) for x in entry["left_keys"]] != expected_left
                or [tuple(x) for x in entry["right_keys"]] != expected_right):
            raise ValueError("geometry cache and topology dataset order differ")
        values = entry["scores"].to(device)
        geometric[batch, :values.shape[0], :values.shape[1]] = values
        filtered_pairs.append(pairs)
        candidate_masks.append(torch.ones_like(values, dtype=torch.bool))
    return filtered_pairs, candidate_masks


def temporal_semantics(output, keys, memory, history_weight, history_size,
                       feature_mode="h_plus_g"):
    """Paper Eq. (14): aggregate prior h only, then concatenate current g."""
    rows = []
    for current_h, current_g, key in zip(output.identity, output.context, keys):
        token = (str(key[0]), int(key[1]), int(key[3]), int(key[4]) if len(key) > 4 else 0)
        history = list(memory.get(token, ()))
        if history:
            previous = torch.stack(history[-history_size:]).mean(dim=0)
            current_h = F.normalize(
                (1.0 - history_weight) * current_h + history_weight * previous, dim=0
            )
        if feature_mode == "h_only":
            current = current_h
        elif feature_mode == "g_only":
            current = current_g
        else:
            current = F.normalize(torch.cat([current_h, current_g]), dim=0)
        rows.append(current)
    if rows:
        return torch.stack(rows)
    dimension = output.identity.shape[-1]
    if feature_mode == "h_plus_g":
        dimension += output.context.shape[-1]
    elif feature_mode == "g_only":
        dimension = output.context.shape[-1]
    return output.identity.new_empty((0, dimension))


def remember_semantics(output, keys, memory, history_size):
    # History contains only the high-order identity representation h.  The
    # relation-state g is always the current-time value in q*.
    for value, key in zip(output.identity, keys):
        token = (str(key[0]), int(key[1]), int(key[3]), int(key[4]) if len(key) > 4 else 0)
        memory[token].append(value)
        while len(memory[token]) > history_size:
            memory[token].popleft()


def detach_memory(memory):
    for token, values in memory.items():
        memory[token] = deque((value.detach() for value in values), maxlen=values.maxlen)


def geometry_for_sample(dataset, index, sample, geometry_windows, device):
    nl, nr = len(sample["left"][0]), len(sample["right"][0])
    if "geometric_score" in sample:
        return (sample["geometric_score"].to(device),
                sample["reverse_geometric_score"].to(device),
                sample["candidate_mask"].to(device).bool(),
                sample["reverse_candidate_mask"].to(device).bool())
    if geometry_windows is None:
        return (torch.zeros(nl, nr, device=device),
                torch.zeros(nr, nl, device=device),
                torch.ones(nl, nr, dtype=torch.bool, device=device),
                torch.ones(nr, nl, dtype=torch.bool, device=device))
    entry = geometry_windows[(str(sample["scene_id"]), int(sample["start"]))]
    window = dataset.windows[index]
    expected_left = [dataset.keys[i] for i in window.left_indices]
    expected_right = [dataset.keys[i] for i in window.right_indices]
    if [tuple(x) for x in entry["left_keys"]] != expected_left or [tuple(x) for x in entry["right_keys"]] != expected_right:
        raise ValueError("geometry cache and topology dataset order differ")
    forward = entry["scores"].to(device)
    mask = entry["candidate_mask"].to(device).bool()
    return forward, forward.T, mask, mask.T


def paper_losses(model, sample, geometric, reverse_geometric,
                 candidate_mask, reverse_candidate_mask,
                 memory, history_weight, history_size,
                 topology_temperature, association_temperature,
                 repair_positive_weight=1.0):
    left = tuple(value.to(geometric.device) for value in sample["left"])
    right = tuple(value.to(geometric.device) for value in sample["right"])
    pairs = sample["positive_pairs"].to(geometric.device)
    if pairs.numel():
        valid_pair_count = int((
            candidate_mask[pairs[:, 0], pairs[:, 1]]
            & reverse_candidate_mask[pairs[:, 1], pairs[:, 0]]
        ).sum().item())
    else:
        valid_pair_count = 0
    _, left_output, right_output = model(left, right, geometric)
    current_topology = model.topology_score(left_output, right_output)
    # Pairs whose current tracker IDs already agree do not alter the official
    # allocator.  Upweight the rare true pairs with different current IDs so
    # training focuses on actual repair opportunities rather than being
    # dominated by trivial confirmations.  The signal is causal tracker state;
    # labels are still used only to identify positive training pairs.
    pair_weights = geometric.new_ones(len(pairs))
    if len(pairs) and repair_positive_weight != 1.0:
        left_ids = torch.as_tensor(
            [int(sample["left_keys"][int(row)][3]) for row in pairs[:, 0]],
            device=geometric.device,
        )
        right_ids = torch.as_tensor(
            [int(sample["right_keys"][int(column)][3]) for column in pairs[:, 1]],
            device=geometric.device,
        )
        pair_weights = torch.where(
            left_ids != right_ids,
            pair_weights.new_full(pair_weights.shape, float(repair_positive_weight)),
            pair_weights,
        )
    topology_loss = association_loss(
        current_topology, pairs, topology_temperature, candidate_mask,
        current_topology.T, reverse_candidate_mask, pair_weights,
    )
    left_temporal = temporal_semantics(
        left_output, sample["left_keys"], memory, history_weight, history_size,
        model.topology.feature_mode,
    )
    right_temporal = temporal_semantics(
        right_output, sample["right_keys"], memory, history_weight, history_size,
        model.topology.feature_mode,
    )
    temporal_topology = 0.5 * (left_temporal @ right_temporal.T + 1.0)
    fused = model.alpha * geometric + (1.0 - model.alpha) * temporal_topology
    reverse_fused = (
        model.alpha * reverse_geometric + (1.0 - model.alpha) * temporal_topology.T
    )
    association = association_loss(
        fused, pairs, association_temperature, candidate_mask,
        reverse_fused, reverse_candidate_mask, pair_weights,
    )
    remember_semantics(left_output, sample["left_keys"], memory, history_size)
    remember_semantics(right_output, sample["right_keys"], memory, history_size)
    return topology_loss, association, fused, reverse_fused, pairs, valid_pair_count


@torch.no_grad()
def evaluate(model, dataset, device, max_windows=0,
             batch_size=8, geometry_windows=None, history_weight=0.5, history_size=4,
             topology_temperature=0.07, association_temperature=0.07):
    model.eval()
    correct = possible = false_matches = missed = 0
    repair_correct = repair_possible = repair_false = repair_missed = 0
    losses = []
    indices = range(min(len(dataset), max_windows)) if max_windows else range(len(dataset))
    indices = list(indices)
    memory = defaultdict(lambda: deque(maxlen=history_size))
    for index in indices:
        sample = dataset[index]
        geometric, reverse_geometric, candidate_mask, reverse_candidate_mask = geometry_for_sample(
            dataset, index, sample, geometry_windows, device
        )
        topo, assoc, scores, reverse_scores, pairs, valid_pair_count = paper_losses(
            model, sample, geometric, reverse_geometric,
            candidate_mask, reverse_candidate_mask,
            memory, history_weight, history_size,
            topology_temperature, association_temperature,
        )
        if valid_pair_count:
            losses.append(float((topo + assoc).item()))
        assign_scores = scores.detach().masked_fill(~candidate_mask, -torch.inf)
        selected = set()
        if assign_scores.numel() and bool(candidate_mask.any()):
            reverse_assign_scores = reverse_scores.detach().masked_fill(
                ~reverse_candidate_mask, -torch.inf
            )
            row_best = assign_scores.argmax(dim=1)
            reverse_best = reverse_assign_scores.argmax(dim=1)
            selected = {
                (row, int(column)) for row, column in enumerate(row_best.tolist())
                if bool(candidate_mask[row, column])
                and bool(reverse_candidate_mask[column, row])
                and int(reverse_best[column]) == row
            }
        truth = {(int(row), int(column)) for row, column in pairs.tolist()}
        window_possible = len(truth); window_correct = len(selected & truth)
        correct += window_correct; false_matches += len(selected - truth)
        possible += window_possible; missed += max(window_possible - window_correct, 0)
        repair_selected = {
            (row, column) for row, column in selected
            if int(sample["left_keys"][row][3]) != int(sample["right_keys"][column][3])
        }
        repair_truth = {
            (row, column) for row, column in truth
            if int(sample["left_keys"][row][3]) != int(sample["right_keys"][column][3])
        }
        repair_correct += len(repair_selected & repair_truth)
        repair_false += len(repair_selected - repair_truth)
        repair_possible += len(repair_truth)
        repair_missed += len(repair_truth - repair_selected)
    denominator = possible + false_matches + missed
    repair_precision = repair_correct / max(repair_correct + repair_false, 1)
    repair_recall = repair_correct / max(repair_possible, 1)
    repair_f0_5 = (1.25 * repair_precision * repair_recall
                   / max(0.25 * repair_precision + repair_recall, 1e-12))
    return {"loss": sum(losses) / max(len(losses), 1), "correct": correct, "possible": possible,
            "false_matches": false_matches, "missed": missed, "mda": correct / max(denominator, 1),
            "repair_correct": repair_correct, "repair_possible": repair_possible,
            "repair_false": repair_false, "repair_missed": repair_missed,
            "repair_precision": repair_precision, "repair_recall": repair_recall,
            "repair_f0_5": repair_f0_5}


def main():
    parser = argparse.ArgumentParser(description="Train complete identity-aware topology enhancement")
    parser.add_argument("--train-cache", required=True)
    parser.add_argument("--val-cache", required=True)
    parser.add_argument("--train-prediction", default="")
    parser.add_argument("--val-prediction", default="")
    parser.add_argument("--train-geometry", default="",
                        help="Frozen official-MIA S_geo cache for training")
    parser.add_argument("--val-geometry", default="",
                        help="Frozen official-MIA S_geo cache for validation")
    parser.add_argument("--xml", default="data/datafull/new_xml")
    parser.add_argument("--out", default="outputs/identity_topology_full/model")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument(
        "--early-stopping-patience", type=int, default=0,
        help="Stop after this many consecutive epochs without validation-selection improvement; 0 disables",
    )
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--optimizer", choices=("sgd", "adamw"), default="sgd")
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Variable-node windows accumulated per optimizer step")
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--patterns", type=int, default=16)
    parser.add_argument("--relation-temperature", type=float, default=0.10)
    parser.add_argument("--pattern-momentum", type=float, default=1.0)
    parser.add_argument("--propagation-layers", type=int, default=1)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--topology-loss-weight", type=float, default=1.0)
    parser.add_argument("--association-loss-weight", type=float, default=1.0)
    parser.add_argument(
        "--repair-positive-weight", type=float, default=1.0,
        help="Ablation weight for true pairs whose current tracker IDs disagree; validation selected 1",
    )
    parser.add_argument("--topology-temperature", type=float, default=0.07)
    parser.add_argument("--association-temperature", type=float, default=0.07)
    parser.add_argument("--history-weight", type=float, default=0.5)
    parser.add_argument("--history-size", type=int, default=4)
    parser.add_argument("--purification", choices=["none", "hard", "continuous"], default="continuous")
    parser.add_argument("--purification-floor", type=float, default=0.05,
                        help="rho0 in continuous purification; compare 0 and 0.05")
    parser.add_argument("--feature-mode", choices=["h_only", "g_only", "h_plus_g"], default="h_plus_g",
                        help="Physically retain h, g, or both feature branches")
    parser.add_argument("--topology-mode", choices=["knn_graph", "graph_learning", "fixed_hypergraph", "dynamic_hypergraph"], default="dynamic_hypergraph")
    parser.add_argument("--max-train-windows", type=int, default=0)
    parser.add_argument("--max-val-windows", type=int, default=0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--allow-nonpaper-appearance", action="store_true",
                        help="Only for ablations/tests: permit appearance features not extracted from frozen MIA FPN")
    parser.add_argument("--allow-missing-geometry", action="store_true",
                        help="Only for ablations/tests: permit zero S_geo")
    parser.add_argument("--allow-controlled-data", action="store_true",
                        help="Only for ablations/tests: permit GT-window instead of predicted-track training")
    parser.add_argument("--allow-method-ablation", action="store_true",
                        help="Only for named ablations: permit removing a paper method component")
    parser.add_argument("--expected-frontend", default="autoassign_bytetrack",
                        choices=["autoassign_bytetrack", "carafe_bytetrack"])
    parser.add_argument(
        "--expected-insertion-point", default="post_mia_output",
        choices=["pre_cross_view_id_allocation", "post_mia_output"],
    )
    parser.add_argument(
        "--expected-pooling", default="assigned_fpn_roi_align_3x3_meanmax",
        choices=["assigned_fpn_roi_align_3x3_meanmax",
                 "per_level_roi_align_1x1_then_equal_fpn_mean"],
    )
    args = parser.parse_args()
    if not 0.0 <= args.alpha <= 1.0:
        parser.error("--alpha must be in [0,1]")
    if not 0.0 <= args.history_weight <= 1.0 or args.history_size < 1:
        parser.error("history weight/size violate the finite-history protocol")
    if args.relation_temperature <= 0 or args.topology_temperature <= 0 or args.association_temperature <= 0:
        parser.error("all paper temperatures must be positive")
    if args.topology_loss_weight <= 0 or args.association_loss_weight <= 0:
        parser.error("both paper losses must have positive weight")
    if args.repair_positive_weight < 1.0:
        parser.error("--repair-positive-weight must be at least 1")
    if not args.allow_method_ablation:
        if (args.topology_mode != "dynamic_hypergraph" or args.purification != "continuous"
                or args.feature_mode != "h_plus_g" or args.propagation_layers != 1
                or args.pattern_momentum != 1.0):
            parser.error("formal training requires the complete paper topology; use --allow-method-ablation only for ablations")
    random.seed(args.seed); torch.manual_seed(args.seed)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    def load_dataset(cache, prediction):
        if prediction:
            return PredictedTopologyFeatureDataset(cache, prediction)
        payload = torch.load(cache, map_location="cpu", weights_only=False)
        if payload.get("metadata", {}).get("representation") == "mia_net_synchronous_frame_targets":
            return SynchronousMIAFrameDataset(payload=payload)
        return TopologyFeatureDataset(cache, args.xml)

    train = load_dataset(args.train_cache, args.train_prediction)
    val = load_dataset(args.val_cache, args.val_prediction)
    synchronous = isinstance(train, SynchronousMIAFrameDataset) and isinstance(val, SynchronousMIAFrameDataset)
    if not args.allow_controlled_data and not synchronous:
        parser.error("paper training requires synchronized per-frame official-MIA targets")
    if not synchronous and not args.allow_missing_geometry and (not args.train_geometry or not args.val_geometry):
        parser.error("tracklet ablations require frozen official-MIA geometry for train and validation")
    if not args.allow_nonpaper_appearance:
        expected = "frozen_official_mia_detector_fpn"
        for split, dataset in (("train", train), ("validation", val)):
            actual = dataset.appearance_metadata.get("appearance_source")
            if actual != expected:
                parser.error(f"{split} appearance_source={actual!r}; expected {expected!r}")
            if dataset.appearance_metadata.get("frontend") != args.expected_frontend:
                parser.error(f"{split} appearance frontend is not {args.expected_frontend}")
            expected_time = ("current_frame_only" if synchronous
                             else "latest_current_observation_only")
            if dataset.appearance_metadata.get("temporal_aggregation") != expected_time:
                parser.error(f"{split} appearance is not the paper's current-time FPN feature")
            report_frontend = dataset.prediction_metadata.get("report", {}).get("frontend")
            if report_frontend != args.expected_frontend:
                parser.error(f"{split} predicted tracks are not from {args.expected_frontend}")
            expected_representation = ("mia_net_synchronous_frame_targets" if synchronous
                                       else "mia_net_persistent_tracks")
            if dataset.prediction_metadata.get("representation") != expected_representation:
                parser.error(f"{split} prediction representation is not {expected_representation}")
            if synchronous:
                metadata = dataset.metadata
                if metadata.get("insertion_point") != args.expected_insertion_point:
                    parser.error(
                        f"{split} insertion point is {metadata.get('insertion_point')!r}; "
                        f"expected {args.expected_insertion_point!r}"
                    )
                if metadata.get("pooling") != args.expected_pooling:
                    parser.error(
                        f"{split} FPN pooling is {metadata.get('pooling')!r}; "
                        f"expected {args.expected_pooling!r}"
                    )
                if metadata.get("topology_time") != "all nodes are targets observed in the same synchronized frame t":
                    parser.error(f"{split} topology nodes are not synchronized at time t")
                if metadata.get("geometry_direction") != "ordered bidirectional view1_to_view2 and view2_to_view1":
                    parser.error(f"{split} embedded geometry is not ordered bidirectional")
                if metadata.get("candidate_rule") != "all synchronized target pairs; geometry is soft and never deletes candidates":
                    parser.error(f"{split} candidate construction is not the paper's soft protocol")
                if metadata.get("geometry_reachability") != "diagnostic only; never used as a candidate mask":
                    parser.error(f"{split} geometry reachability still acts as a hard candidate filter")
                if metadata.get("geometry_scales") != {"new": 80.0, "existing": 50.0}:
                    parser.error(f"{split} geometry does not use MIA 80/50 state scales")
                if metadata.get("geometry_state") != "official MIA new iff current ID exceeds previous committed maximum ID":
                    parser.error(f"{split} geometry does not use the official MIA new/old state machine")
                if not metadata.get("truth_labels_present"):
                    parser.error(f"{split} cache lacks supervision labels")
    train_geometry_payload = (torch.load(args.train_geometry, map_location="cpu", weights_only=False)
                              if args.train_geometry else None)
    val_geometry_payload = (torch.load(args.val_geometry, map_location="cpu", weights_only=False)
                            if args.val_geometry else None)
    if not synchronous and not args.allow_missing_geometry:
        for split, payload in (("train", train_geometry_payload), ("validation", val_geometry_payload)):
            protocol = payload.get("protocol", {})
            if protocol.get("direction") != "view1_to_view2":
                parser.error(f"{split} geometry is not the paper's directed MIA score")
            if protocol.get("candidate_rule") != "temporal co-visibility; no geometric-score threshold":
                parser.error(f"{split} geometry does not use the paper's soft candidate protocol")
            if float(protocol.get("new_target_scale", -1)) != 80.0 or float(protocol.get("existing_target_scale", -1)) != 50.0:
                parser.error(f"{split} geometry does not use official MIA 80/50 state scales")
    train_geometry = train_geometry_payload["windows"] if train_geometry_payload else None
    val_geometry = val_geometry_payload["windows"] if val_geometry_payload else None
    model = TopologyEnhancedAssociation(
        train.appearance_dim, alpha=args.alpha,
        hidden_dim=args.hidden_dim, patterns=args.patterns,
        propagation_layers=args.propagation_layers, purification=args.purification,
        topology_mode=args.topology_mode, pattern_momentum=args.pattern_momentum,
        feature_mode=args.feature_mode, purification_floor=args.purification_floor,
        temperature=args.relation_temperature,
    ).to(args.device)
    if args.optimizer == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(), lr=args.lr, momentum=args.momentum,
            weight_decay=args.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[8], gamma=0.1)
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    history = []; best_selection = (-1.0, -1.0); epochs_without_improvement = 0
    for epoch in range(1, args.epochs + 1):
        # Preserve causal scene/start ordering for the finite history in Eq. (13).
        model.train(); indices = list(range(len(train)))
        if args.max_train_windows: indices = indices[:args.max_train_windows]
        totals = {"loss": 0.0, "association": 0.0, "topology": 0.0}
        supervised_windows = 0
        memory = defaultdict(lambda: deque(maxlen=args.history_size))
        batch_losses = []
        optimizer.zero_grad(set_to_none=True)
        for position, index in enumerate(indices, 1):
            sample = train[index]
            geometric, reverse_geometric, candidate_mask, reverse_candidate_mask = geometry_for_sample(
                train, index, sample, train_geometry, args.device
            )
            topology_loss, association, _, _, _, valid_pair_count = paper_losses(
                model, sample, geometric, reverse_geometric,
                candidate_mask, reverse_candidate_mask, memory,
                args.history_weight, args.history_size,
                args.topology_temperature, args.association_temperature,
                args.repair_positive_weight,
            )
            loss = (args.topology_loss_weight * topology_loss
                    + args.association_loss_weight * association)
            if valid_pair_count:
                batch_losses.append(loss)
                supervised_windows += 1
                totals["loss"] += float(loss.item())
                totals["association"] += float(association.item())
                totals["topology"] += float(topology_loss.item())
            if len(batch_losses) == args.batch_size or (position == len(indices) and batch_losses):
                torch.stack(batch_losses).mean().backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step(); optimizer.zero_grad(set_to_none=True)
                detach_memory(memory); batch_losses = []
        train_metrics = {key: value / max(supervised_windows, 1) for key, value in totals.items()}
        train_metrics["supervised_windows"] = supervised_windows
        val_metrics = evaluate(
            model, val, args.device, args.max_val_windows, args.batch_size, val_geometry,
            args.history_weight, args.history_size,
            args.topology_temperature, args.association_temperature,
        )
        row = {"epoch": epoch, "train": train_metrics, "val": val_metrics}; history.append(row)
        print(json.dumps(row), flush=True)
        state = {"model": model.state_dict(), "config": vars(args), "appearance_dim": train.appearance_dim,
                 "epoch": epoch, "val": val_metrics}
        torch.save(state, out / "last_checkpoint.pt")
        selection = (val_metrics["repair_f0_5"], val_metrics["mda"])
        if selection > best_selection:
            best_selection = selection
            epochs_without_improvement = 0
            torch.save(state, out / "best_checkpoint.pt")
        else:
            epochs_without_improvement += 1
        (out / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        scheduler.step()
        if (args.early_stopping_patience > 0
                and epochs_without_improvement >= args.early_stopping_patience):
            print(json.dumps({
                "early_stopped": True,
                "epoch": epoch,
                "patience": args.early_stopping_patience,
                "best_selection": best_selection,
            }), flush=True)
            break
    (out / "config.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
