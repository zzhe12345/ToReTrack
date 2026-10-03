from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F


@dataclass
class TopologyOutput:
    nodes: torch.Tensor
    memberships: torch.Tensor
    reliability: torch.Tensor
    identity: torch.Tensor
    context: torch.Tensor
    semantic: torch.Tensor


def _mlp(input_dim: int, hidden_dim: int, output_dim: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.LayerNorm(hidden_dim),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, output_dim),
    )


class IdentityAwareTopology(nn.Module):
    """Executable form of the complete identity-aware topology method.

    Appearance, normalized box state, and trajectory dynamics are independently
    projected and added. Learnable relation primitives define a soft incidence
    matrix. A reliability predictor purifies memberships, one higher-order
    propagation layer produces h, and the purified primitive sum produces g.
    """

    def __init__(
        self,
        appearance_dim: int,
        spatial_dim: int = 4,
        trajectory_dim: int = 5,
        hidden_dim: int = 256,
        patterns: int = 16,
        propagation_layers: int = 1,
        temperature: float = 0.10,
        dropout: float = 0.10,
        purification: str = "continuous",
        hard_threshold: float = 0.50,
        topology_mode: str = "dynamic_hypergraph",
        fixed_k: int = 5,
        pattern_momentum: float = 1.0,
        feature_mode: str = "h_plus_g",
        purification_floor: float = 0.05,
    ) -> None:
        super().__init__()
        if purification not in {"none", "hard", "continuous"}:
            raise ValueError(f"unsupported purification: {purification}")
        if topology_mode not in {"knn_graph", "graph_learning", "fixed_hypergraph", "dynamic_hypergraph"}:
            raise ValueError(f"unsupported topology mode: {topology_mode}")
        if propagation_layers != 1:
            raise ValueError("the paper defines exactly one higher-order propagation matrix")
        if pattern_momentum != 1.0:
            raise ValueError("paper-aligned relation prototypes are global learned parameters, not scene-adapted")
        if feature_mode not in {"h_only", "g_only", "h_plus_g"}:
            raise ValueError(f"unsupported feature mode: {feature_mode}")
        if not 0.0 <= purification_floor <= 1.0:
            raise ValueError("purification_floor must be between 0 and 1")
        self.hidden_dim = hidden_dim
        self.patterns_count = patterns
        self.temperature = temperature
        self.purification = purification
        self.hard_threshold = hard_threshold
        self.topology_mode = topology_mode
        self.fixed_k = fixed_k
        self.pattern_momentum = pattern_momentum
        self.feature_mode = feature_mode
        self.purification_floor = purification_floor
        # Eq. (3): three independent projections into the same d-dimensional
        # relation space, followed by element-wise addition and L2
        # normalization.  Concatenation appears only inside the trajectory
        # input [f_traj || delta] and later in the parameter-free [h || g].
        self.appearance_projection = nn.Linear(appearance_dim, hidden_dim, bias=False)
        self.spatial_projection = nn.Linear(spatial_dim, hidden_dim, bias=False)
        self.trajectory_projection = nn.Linear(trajectory_dim, hidden_dim, bias=False)
        self.patterns = nn.Parameter(torch.randn(patterns, hidden_dim) * 0.02)
        # Eq. (6): a two-layer ReLU MLP over node, prototype and their
        # element-wise interaction.  It predicts membership reliability only.
        self.reliability = nn.Sequential(
            nn.Linear(hidden_dim * 3 + 1, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1)
        )
        self.graph_query = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.graph_key = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.propagation = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.identity_head = None
        self.context_head = None

    def _purify(self, memberships: torch.Tensor, reliability: torch.Tensor) -> torch.Tensor:
        if self.purification == "none":
            purified = memberships
        elif self.purification == "hard":
            mask = (reliability >= self.hard_threshold).to(memberships.dtype)
            purified = memberships * mask
        else:
            # Γ(H, τ): continuous confidence shrinkage with an identity path.
            rho0 = self.purification_floor
            purified = memberships * (rho0 + (1.0 - rho0) * reliability)
        # The paper explicitly forbids row renormalization here: absolute
        # structural reliability must survive into the degree matrices.
        return purified

    def _semantics(self, identity: torch.Tensor, relation_state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Eq. (9): g_i is the purified membership-weighted relation-prototype
        # state.  The normalized incidence row itself is not the paper's g_i.
        context = F.normalize(relation_state, dim=-1)
        if self.feature_mode == "h_only":
            semantic = identity
        elif self.feature_mode == "g_only":
            semantic = context
        else:
            # Eq. (10): parameter-free normalized concatenation of h and g.
            semantic = F.normalize(torch.cat([identity, context], dim=-1), dim=-1)
        return identity, context, semantic

    def encode(self, appearance: torch.Tensor, spatial: torch.Tensor, trajectory: torch.Tensor) -> TopologyOutput:
        nodes = F.normalize(
            self.appearance_projection(appearance)
            + self.spatial_projection(spatial)
            + self.trajectory_projection(trajectory),
            dim=-1,
        )
        patterns = F.normalize(self.patterns, dim=-1)
        if self.topology_mode == "dynamic_hypergraph":
            memberships = F.softmax(nodes @ patterns.T / self.temperature, dim=-1)
            node_expanded = nodes[:, None, :].expand(-1, self.patterns_count, -1)
            pattern_expanded = self.patterns[None, :, :].expand(len(nodes), -1, -1)
            reliability_input = torch.cat([
                node_expanded, pattern_expanded, node_expanded * pattern_expanded,
                memberships[..., None],
            ], dim=-1)
            reliability = torch.sigmoid(self.reliability(reliability_input).squeeze(-1))
            incidence = self._purify(memberships, reliability)
            edge_degree = incidence.sum(dim=0).clamp_min(1e-6)
            vertex_degree = incidence.sum(dim=1).clamp_min(1e-6)
            normalized = vertex_degree.rsqrt()[:, None] * incidence * edge_degree.rsqrt()[None, :]
            propagation = normalized @ normalized.T
            context_base = incidence
            # Eq. (9) sums the learnable relation primitives m_k themselves;
            # cosine normalization is used only to form H in Eq. (4).
            relation_state = incidence @ self.patterns
        else:
            n = len(nodes)
            similarity = nodes @ nodes.T
            if self.topology_mode == "graph_learning":
                query = self.graph_query(nodes)
                key = self.graph_key(nodes)
                reliability = torch.sigmoid(query @ key.T / self.hidden_dim ** 0.5)
                memberships = F.softmax(similarity / self.temperature, dim=1)
                propagation = self._purify(memberships, reliability)
            else:
                neighbor_count = min(self.fixed_k, max(n - 1, 1))
                scores = similarity.masked_fill(torch.eye(n, dtype=torch.bool, device=nodes.device), -2.0)
                neighbors = scores.topk(min(neighbor_count, n), dim=1).indices
                incidence = nodes.new_zeros(n, n)
                incidence.scatter_(1, neighbors, 1.0)
                incidence.fill_diagonal_(1.0)
                memberships = incidence / incidence.sum(dim=1, keepdim=True).clamp_min(1.0)
                reliability = torch.ones_like(memberships)
                if self.topology_mode == "fixed_hypergraph":
                    edge_degree = incidence.sum(dim=0).clamp_min(1e-6)
                    vertex_degree = incidence.sum(dim=1).clamp_min(1e-6)
                    normalized = vertex_degree.rsqrt()[:, None] * incidence * edge_degree.rsqrt()[None, :]
                    propagation = normalized @ normalized.T
                else:
                    propagation = memberships
            context_base = propagation @ nodes
            relation_state = context_base
        identity = F.normalize(nodes + self.propagation(propagation @ nodes), dim=-1)
        identity, context, semantic = self._semantics(identity, relation_state)
        return TopologyOutput(nodes, context_base, reliability, identity, context, semantic)

    def encode_batched(
        self, appearance: torch.Tensor, spatial: torch.Tensor,
        trajectory: torch.Tensor, mask: torch.Tensor,
    ) -> TopologyOutput:
        """Padded, masked version of :meth:`encode` for full experiment runs."""
        nodes = F.normalize(
            self.appearance_projection(appearance)
            + self.spatial_projection(spatial)
            + self.trajectory_projection(trajectory),
            dim=-1,
        )
        nodes = nodes * mask[..., None]
        base_patterns = F.normalize(self.patterns, dim=-1)
        batch, count, _ = nodes.shape
        if self.topology_mode == "dynamic_hypergraph":
            memberships = F.softmax(
                torch.einsum("bnh,kh->bnk", nodes, base_patterns) / self.temperature, dim=-1
            ) * mask[..., None]
            patterns = base_patterns[None].expand(batch, -1, -1)
            raw_patterns = self.patterns[None].expand(batch, -1, -1)
            node_expanded = nodes[:, :, None, :].expand(-1, -1, self.patterns_count, -1)
            pattern_expanded = raw_patterns[:, None, :, :].expand(-1, count, -1, -1)
            reliability_input = torch.cat([
                node_expanded, pattern_expanded, node_expanded * pattern_expanded,
                memberships[..., None],
            ], dim=-1)
            reliability = torch.sigmoid(self.reliability(reliability_input).squeeze(-1)) * mask[..., None]
            incidence = self._purify(memberships, reliability) * mask[..., None]
            edge_degree = incidence.sum(dim=1).clamp_min(1e-6)
            vertex_degree = incidence.sum(dim=2).clamp_min(1e-6)
            normalized = vertex_degree.rsqrt()[..., None] * incidence * edge_degree.rsqrt()[:, None, :]
            propagation = torch.bmm(normalized, normalized.transpose(1, 2))
            context_base = incidence
            relation_state = torch.bmm(
                incidence, self.patterns[None].expand(batch, -1, -1)
            )
        else:
            similarity = torch.bmm(nodes, nodes.transpose(1, 2))
            valid_pairs = mask[:, :, None] & mask[:, None, :]
            if self.topology_mode == "graph_learning":
                query = self.graph_query(nodes)
                key = self.graph_key(nodes)
                reliability = torch.sigmoid(
                    torch.bmm(query, key.transpose(1, 2)) / self.hidden_dim ** 0.5
                ) * valid_pairs
                masked_similarity = similarity.masked_fill(~mask[:, None, :], -1e4)
                memberships = F.softmax(masked_similarity / self.temperature, dim=2) * mask[:, :, None]
                propagation = self._purify(memberships, reliability) * valid_pairs
            else:
                k = min(self.fixed_k, max(count - 1, 1))
                eye = torch.eye(count, dtype=torch.bool, device=nodes.device)[None]
                scores = similarity.masked_fill(~valid_pairs | eye, -1e4)
                neighbors = scores.topk(k, dim=2).indices
                incidence = nodes.new_zeros(batch, count, count)
                incidence.scatter_(2, neighbors, 1.0)
                incidence = incidence * valid_pairs
                incidence = incidence + torch.diag_embed(mask.to(nodes.dtype))
                memberships = incidence / incidence.sum(dim=2, keepdim=True).clamp_min(1.0)
                reliability = valid_pairs.to(nodes.dtype)
                if self.topology_mode == "fixed_hypergraph":
                    edge_degree = incidence.sum(dim=1).clamp_min(1e-6)
                    vertex_degree = incidence.sum(dim=2).clamp_min(1e-6)
                    normalized = vertex_degree.rsqrt()[..., None] * incidence * edge_degree.rsqrt()[:, None, :]
                    propagation = torch.bmm(normalized, normalized.transpose(1, 2))
                else:
                    propagation = memberships
            context_base = torch.bmm(propagation, nodes)
            relation_state = context_base
        identity = F.normalize(nodes + self.propagation(torch.bmm(propagation, nodes)), dim=-1) * mask[..., None]
        identity, context, semantic = self._semantics(identity, relation_state)
        context = context * mask[..., None]
        semantic = semantic * mask[..., None]
        return TopologyOutput(nodes, context_base, reliability, identity, context, semantic)


class TopologyEnhancedAssociation(nn.Module):
    def __init__(self, appearance_dim: int, alpha: float = 0.50, context_weight: float = 0.50, **kwargs) -> None:
        super().__init__()
        self.topology = IdentityAwareTopology(appearance_dim, **kwargs)
        self.alpha = alpha
        self.context_weight = context_weight

    def topology_score(self, left: TopologyOutput, right: TopologyOutput) -> torch.Tensor:
        return left.semantic @ right.semantic.T

    def topology_score_batched(self, left: TopologyOutput, right: TopologyOutput) -> torch.Tensor:
        return torch.bmm(left.semantic, right.semantic.transpose(1, 2))

    def forward(
        self,
        left: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        right: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        geometric_score: torch.Tensor,
    ) -> tuple[torch.Tensor, TopologyOutput, TopologyOutput]:
        left_output = self.topology.encode(*left)
        right_output = self.topology.encode(*right)
        topology = self.topology_score(left_output, right_output)
        topology = 0.5 * (topology + 1.0)
        # Eq. (12): alpha is the geometry-evidence weight.
        score = self.alpha * geometric_score + (1.0 - self.alpha) * topology
        return score, left_output, right_output

    def forward_batched(self, left, right, left_mask, right_mask, geometric_score):
        left_output = self.topology.encode_batched(*left, left_mask)
        right_output = self.topology.encode_batched(*right, right_mask)
        topology = self.topology_score_batched(left_output, right_output)
        topology = 0.5 * (topology + 1.0)
        score = self.alpha * geometric_score + (1.0 - self.alpha) * topology
        return score, left_output, right_output


def supervised_alignment_loss(
    left: TopologyOutput,
    right: TopologyOutput,
    positive_pairs: torch.Tensor,
    identity_weight: float = 1.0,
    context_weight: float = 0.5,
    temperature: float = 0.07,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Symmetric cross-view identity and structural-context alignment."""
    if positive_pairs.numel() == 0:
        zero = left.identity.sum() * 0.0
        return zero, {"identity": zero, "context": zero}
    li, ri = positive_pairs[:, 0].long(), positive_pairs[:, 1].long()
    target_lr = ri
    target_rl = li

    def symmetric(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        logits = a @ b.T / temperature
        return 0.5 * (
            F.cross_entropy(logits[li], target_lr) + F.cross_entropy(logits.T[ri], target_rl)
        )

    semantic = symmetric(left.semantic, right.semantic)
    zero = semantic.detach() * 0.0
    return semantic, {"identity": semantic, "context": zero}


def membership_regularization(output: TopologyOutput) -> torch.Tensor:
    """Prevent pattern collapse while encouraging selective memberships."""
    usage = output.memberships.mean(dim=0)
    balance = torch.sum(usage * torch.log(usage.clamp_min(1e-8))) + torch.log(
        usage.new_tensor(float(len(usage)))
    )
    entropy = -(output.memberships * torch.log(output.memberships.clamp_min(1e-8))).sum(dim=1).mean()
    return balance + 0.01 * entropy
