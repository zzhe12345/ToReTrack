from __future__ import annotations

"""Causal high-precision gate for deciding whether to overwrite an MIA ID."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn


FEATURE_NAMES = (
    "fused_bidirectional_mean",
    "fused_bidirectional_min",
    "fused_forward_margin",
    "fused_reverse_margin",
    "geometry_forward",
    "geometry_reverse",
    "temporal_topology",
    "current_topology",
    "appearance_similarity",
    "appearance_mutual_best",
    "geometry_mutual_best",
    "topology_mutual_best",
    "branch_agreement_fraction",
    "log_forward_candidates",
    "log_reverse_candidates",
    "left_track_age",
    "right_track_age",
    "left_history_valid",
    "right_history_valid",
    "left_new_target",
    "right_new_target",
    "both_stable_different_ids",
    "left_id_conflict",
    "right_id_conflict",
    "left_box_scale",
    "right_box_scale",
    "left_border_margin",
    "right_border_margin",
    "pair_consecutive_support",
    "pair_fused_score_ema",
    "pair_score_stability",
    "pair_fused_score_mean",
    "pair_fused_score_min",
    "pair_fused_score_std",
    "pair_geometry_ema",
    "pair_topology_ema",
    "pair_appearance_ema",
    "pair_branch_agreement_ema",
)


@dataclass(frozen=True)
class MutualCandidate:
    row: int
    column: int
    forward_margin: float
    reverse_margin: float
    forward_candidates: int
    reverse_candidates: int


class RepairGate(nn.Module):
    def __init__(self, input_dim: int = len(FEATURE_NAMES), hidden_dim: int = 64,
                 dropout: float = 0.10) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features).squeeze(-1)


def _second_margin(values: np.ndarray, valid: np.ndarray, selected: int) -> tuple[float, int]:
    indices = np.flatnonzero(valid)
    others = indices[indices != selected]
    # A singleton still needs an absolute score/gate decision.  Use the full
    # unit interval as its relative margin and expose candidate count as a
    # separate feature instead of treating the margin as infinite.
    margin = float(values[selected] - values[others].max()) if len(others) else 1.0
    return margin, int(len(indices))


def mutual_candidates(forward: np.ndarray, reverse: np.ndarray,
                      mask: np.ndarray, reverse_mask: np.ndarray) -> list[MutualCandidate]:
    return bidirectional_topk_candidates(
        forward, reverse, mask, reverse_mask, topk=1,
    )


def bidirectional_topk_candidates(
    forward: np.ndarray,
    reverse: np.ndarray,
    mask: np.ndarray,
    reverse_mask: np.ndarray,
    topk: int = 1,
) -> list[MutualCandidate]:
    """Return pairs ranked in the Top-K in both ordered view directions.

    ``topk=1`` is exactly the original mutual-best protocol.  For K > 1 the
    selected-vs-best-alternative margins are deliberately signed: a negative
    margin tells the repair gate how far a non-best proposal trails the row or
    column winner.  Candidate expansion therefore increases recall without
    silently pretending that every expanded edge is a confident match.
    """
    if topk < 1:
        raise ValueError("topk must be at least one")
    if (forward.size == 0 or reverse.size == 0 or not mask.any()
            or not reverse_mask.any()):
        return []
    masked = np.where(mask, forward, -np.inf)
    reverse_masked = np.where(reverse_mask, reverse, -np.inf)
    forward_rank = np.full(mask.shape, np.iinfo(np.int32).max, dtype=np.int32)
    reverse_rank = np.full(reverse_mask.shape, np.iinfo(np.int32).max, dtype=np.int32)
    for row in range(masked.shape[0]):
        columns = np.flatnonzero(mask[row])
        order = columns[np.argsort(-masked[row, columns], kind="stable")]
        forward_rank[row, order] = np.arange(1, len(order) + 1)
    for column in range(reverse_masked.shape[0]):
        rows = np.flatnonzero(reverse_mask[column])
        order = rows[np.argsort(-reverse_masked[column, rows], kind="stable")]
        reverse_rank[column, order] = np.arange(1, len(order) + 1)
    result = []
    selected = (
        mask & reverse_mask.T
        & (forward_rank <= int(topk))
        & (reverse_rank.T <= int(topk))
    )
    for row, column in np.argwhere(selected):
        row, column = int(row), int(column)
        if not mask[row, column] or not reverse_mask[column, row]:
            continue
        forward_margin, forward_count = _second_margin(
            forward[row], mask[row], column
        )
        reverse_margin, reverse_count = _second_margin(
            reverse[column], reverse_mask[column], row
        )
        result.append(MutualCandidate(
            row, column, forward_margin, reverse_margin,
            forward_count, reverse_count,
        ))
    return sorted(result, key=lambda item: (item.row, item.column))


def either_direction_topk_candidates(
    forward: np.ndarray,
    reverse: np.ndarray,
    mask: np.ndarray,
    reverse_mask: np.ndarray,
    topk: int = 1,
) -> list[MutualCandidate]:
    """Return the union of ordered-view Top-K proposals.

    This is a controlled high-recall generator for the repair gate.  Unlike a
    dense Cartesian candidate expansion, K=1 adds only row or column winners.
    The signed margins still identify whether an edge is mutual-best, a
    one-sided winner, or lower ranked in both directions.
    """
    if topk < 1:
        raise ValueError("topk must be at least one")
    if (forward.size == 0 or reverse.size == 0 or not mask.any()
            or not reverse_mask.any()):
        return []
    forward_rank = np.full(mask.shape, np.iinfo(np.int32).max, dtype=np.int32)
    reverse_rank = np.full(reverse_mask.shape, np.iinfo(np.int32).max, dtype=np.int32)
    for row in range(forward.shape[0]):
        columns = np.flatnonzero(mask[row])
        order = columns[np.argsort(-forward[row, columns], kind="stable")]
        forward_rank[row, order] = np.arange(1, len(order) + 1)
    for column in range(reverse.shape[0]):
        rows = np.flatnonzero(reverse_mask[column])
        order = rows[np.argsort(-reverse[column, rows], kind="stable")]
        reverse_rank[column, order] = np.arange(1, len(order) + 1)
    selected = (
        mask & reverse_mask.T
        & ((forward_rank <= int(topk)) | (reverse_rank.T <= int(topk)))
    )
    result = []
    for row, column in np.argwhere(selected):
        row, column = int(row), int(column)
        forward_margin, forward_count = _second_margin(
            forward[row], mask[row], column
        )
        reverse_margin, reverse_count = _second_margin(
            reverse[column], reverse_mask[column], row
        )
        result.append(MutualCandidate(
            row, column, forward_margin, reverse_margin,
            forward_count, reverse_count,
        ))
    return sorted(result, key=lambda item: (item.row, item.column))


def _is_mutual(forward: np.ndarray, reverse: np.ndarray,
               mask: np.ndarray, reverse_mask: np.ndarray,
               row: int, column: int) -> bool:
    if not mask[row, column] or not reverse_mask[column, row]:
        return False
    row_values = np.where(mask[row], forward[row], -np.inf)
    reverse_values = np.where(reverse_mask[column], reverse[column], -np.inf)
    return int(row_values.argmax()) == column and int(reverse_values.argmax()) == row


def update_track_ages(keys: list[tuple], frame: int, state: dict) -> np.ndarray:
    ages = []
    for key in keys:
        token = (str(key[0]), int(key[1]), int(key[3]), int(key[4]) if len(key) > 4 else 0)
        previous_frame, previous_age = state.get(token, (-2, 0))
        age = previous_age + 1 if previous_frame == frame - 1 else 1
        state[token] = (frame, age)
        ages.append(age)
    return np.asarray(ages, dtype=np.int64)


def update_pair_support(state: dict, key: tuple, frame: int) -> int:
    """Return causal consecutive-frame support for one proposed ID repair."""
    previous_frame, previous_support = state.get(key, (-2, 0))
    support = previous_support + 1 if previous_frame == frame - 1 else 1
    state[key] = (frame, support)
    stale = [token for token, (last_frame, _count) in state.items()
             if last_frame < frame - 1]
    for token in stale:
        state.pop(token, None)
    return support


def update_pair_evidence(state: dict, key: tuple, frame: int,
                         score: float) -> tuple[int, float, float]:
    """Update causal persistence, score EMA, and one-step stability."""
    previous = state.get(key)
    consecutive = previous is not None and previous[0] == frame - 1
    if consecutive:
        support = int(previous[1]) + 1
        ema = 0.5 * float(previous[2]) + 0.5 * float(score)
        stability = 1.0 - min(abs(float(score) - float(previous[3])), 1.0)
    else:
        support, ema, stability = 1, float(score), 0.0
    state[key] = (frame, support, ema, float(score))
    stale = [token for token, values in state.items() if values[0] < frame - 1]
    for token in stale:
        state.pop(token, None)
    return support, ema, stability


def update_pair_multiframe_evidence(
    state: dict, key: tuple, frame: int, fused_score: float,
    geometry_score: float, topology_score: float,
    appearance_score: float, branch_agreement: float,
) -> tuple[float, ...]:
    """Causal running state used by the trajectory-link utility gate."""
    previous = state.get(key)
    consecutive = previous is not None and previous[0] == frame - 1
    if consecutive:
        support = int(previous[1]) + 1
        fused_ema = 0.5 * float(previous[2]) + 0.5 * float(fused_score)
        stability = 1.0 - min(abs(float(fused_score) - float(previous[3])), 1.0)
        old_mean, old_m2 = float(previous[4]), float(previous[5])
        delta = float(fused_score) - old_mean
        fused_mean = old_mean + delta / support
        fused_m2 = old_m2 + delta * (float(fused_score) - fused_mean)
        fused_min = min(float(previous[6]), float(fused_score))
        geometry_ema = 0.5 * float(previous[7]) + 0.5 * float(geometry_score)
        topology_ema = 0.5 * float(previous[8]) + 0.5 * float(topology_score)
        appearance_ema = 0.5 * float(previous[9]) + 0.5 * float(appearance_score)
        agreement_ema = 0.5 * float(previous[10]) + 0.5 * float(branch_agreement)
    else:
        support = 1
        fused_ema = fused_mean = fused_min = float(fused_score)
        fused_m2 = 0.0
        stability = 0.0
        geometry_ema = float(geometry_score)
        topology_ema = float(topology_score)
        appearance_ema = float(appearance_score)
        agreement_ema = float(branch_agreement)
    fused_std = float(np.sqrt(max(fused_m2 / max(support - 1, 1), 0.0)))
    state[key] = (
        frame, support, fused_ema, float(fused_score), fused_mean, fused_m2,
        fused_min, geometry_ema, topology_ema, appearance_ema, agreement_ema,
    )
    stale = [token for token, values in state.items() if values[0] < frame - 1]
    for token in stale:
        state.pop(token, None)
    return (
        support, fused_ema, stability, fused_mean, fused_min, fused_std,
        geometry_ema, topology_ema, appearance_ema, agreement_ema,
    )


def candidate_branch_agreement(
    appearance: np.ndarray, geometry: np.ndarray,
    reverse_geometry: np.ndarray, topology: np.ndarray,
    mask: np.ndarray, reverse_mask: np.ndarray, row: int, column: int,
) -> float:
    appearance_mutual = _is_mutual(
        appearance, appearance.T, mask, reverse_mask, row, column
    )
    geometry_mutual = _is_mutual(
        geometry, reverse_geometry, mask, reverse_mask, row, column
    )
    topology_mutual = _is_mutual(
        topology, topology.T, mask, reverse_mask, row, column
    )
    return float(appearance_mutual + geometry_mutual + topology_mutual) / 3.0


def candidate_feature(
    candidate: MutualCandidate,
    forward_fused: np.ndarray,
    reverse_fused: np.ndarray,
    geometry: np.ndarray,
    reverse_geometry: np.ndarray,
    temporal_topology: np.ndarray,
    current_topology: np.ndarray,
    appearance: np.ndarray,
    mask: np.ndarray,
    reverse_mask: np.ndarray,
    left_keys: list[tuple],
    right_keys: list[tuple],
    left_new: np.ndarray,
    right_new: np.ndarray,
    left_spatial: np.ndarray,
    right_spatial: np.ndarray,
    left_trajectory: np.ndarray,
    right_trajectory: np.ndarray,
    left_ages: np.ndarray,
    right_ages: np.ndarray,
    pair_support: int = 1,
    pair_score_ema: float | None = None,
    pair_score_stability: float = 0.0,
    pair_score_mean: float | None = None,
    pair_score_min: float | None = None,
    pair_score_std: float = 0.0,
    pair_geometry_ema: float | None = None,
    pair_topology_ema: float | None = None,
    pair_appearance_ema: float | None = None,
    pair_branch_agreement_ema: float | None = None,
) -> np.ndarray:
    row, column = candidate.row, candidate.column
    appearance_mutual = _is_mutual(
        appearance, appearance.T, mask, reverse_mask, row, column
    )
    geometry_mutual = _is_mutual(
        geometry, reverse_geometry, mask, reverse_mask, row, column
    )
    topology_mutual = _is_mutual(
        temporal_topology, temporal_topology.T, mask, reverse_mask, row, column
    )
    agreement = float(appearance_mutual + geometry_mutual + topology_mutual) / 3.0
    left_id = int(left_keys[row][3]); right_id = int(right_keys[column][3])
    left_ids = [int(key[3]) for key in left_keys]
    right_ids = [int(key[3]) for key in right_keys]
    left_conflict = any(value == left_id for index, value in enumerate(right_ids) if index != column)
    right_conflict = any(value == right_id for index, value in enumerate(left_ids) if index != row)
    both_stable = bool(not left_new[row] and not right_new[column] and left_id != right_id)
    left_state, right_state = left_spatial[row], right_spatial[column]
    left_border = max(min(left_state[0], 1.0 - left_state[0],
                          left_state[1], 1.0 - left_state[1]), 0.0)
    right_border = max(min(right_state[0], 1.0 - right_state[0],
                           right_state[1], 1.0 - right_state[1]), 0.0)
    age_scale = np.log1p(20.0)
    fused_value = 0.5 * (
        forward_fused[row, column] + reverse_fused[column, row]
    )
    geometry_value = 0.5 * (
        geometry[row, column] + reverse_geometry[column, row]
    )
    values = (
        fused_value,
        min(forward_fused[row, column], reverse_fused[column, row]),
        candidate.forward_margin,
        candidate.reverse_margin,
        geometry[row, column],
        reverse_geometry[column, row],
        temporal_topology[row, column],
        current_topology[row, column],
        appearance[row, column],
        float(appearance_mutual),
        float(geometry_mutual),
        float(topology_mutual),
        agreement,
        np.log1p(candidate.forward_candidates) / 5.0,
        np.log1p(candidate.reverse_candidates) / 5.0,
        min(np.log1p(left_ages[row]) / age_scale, 1.0),
        min(np.log1p(right_ages[column]) / age_scale, 1.0),
        float(left_trajectory[row, 4] > 0.5),
        float(right_trajectory[column, 4] > 0.5),
        float(left_new[row]),
        float(right_new[column]),
        float(both_stable),
        float(left_conflict),
        float(right_conflict),
        float(np.sqrt(max(left_state[2] * left_state[3], 0.0))),
        float(np.sqrt(max(right_state[2] * right_state[3], 0.0))),
        float(left_border),
        float(right_border),
        min(np.log1p(max(pair_support, 1)) / np.log1p(10.0), 1.0),
        float(fused_value if pair_score_ema is None else pair_score_ema),
        float(pair_score_stability),
        float(fused_value if pair_score_mean is None else pair_score_mean),
        float(fused_value if pair_score_min is None else pair_score_min),
        float(pair_score_std),
        float(geometry_value if pair_geometry_ema is None else pair_geometry_ema),
        float(temporal_topology[row, column]
              if pair_topology_ema is None else pair_topology_ema),
        float(appearance[row, column]
              if pair_appearance_ema is None else pair_appearance_ema),
        float(agreement if pair_branch_agreement_ema is None
              else pair_branch_agreement_ema),
    )
    return np.asarray(values, dtype=np.float32)


def load_repair_gate(path: str | Path, device: str):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    checkpoint_names = tuple(payload.get("feature_names", ()))
    if checkpoint_names != FEATURE_NAMES[:len(checkpoint_names)]:
        raise ValueError("repair gate feature protocol differs from the runtime")
    if not checkpoint_names:
        raise ValueError("repair gate checkpoint has no feature protocol")
    model = RepairGate(
        len(checkpoint_names), int(payload.get("hidden_dim", 64)),
        float(payload.get("dropout", 0.10)),
    )
    model.load_state_dict(payload["model"])
    model.to(device).eval()
    mean = torch.as_tensor(payload["feature_mean"], dtype=torch.float32, device=device)
    scale = torch.as_tensor(payload["feature_scale"], dtype=torch.float32, device=device)
    return model, mean, scale, payload
