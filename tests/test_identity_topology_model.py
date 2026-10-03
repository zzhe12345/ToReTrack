import torch
from collections import defaultdict, deque

from uav_tracking.identity_topology_model import (
    TopologyEnhancedAssociation,
    membership_regularization,
    supervised_alignment_loss,
)
from evaluate_identity_topology import competition_assignment
from train_identity_topology import association_loss, temporal_semantics


def _view(nodes: int, appearance_dim: int = 32):
    return torch.randn(nodes, appearance_dim), torch.randn(nodes, 4), torch.randn(nodes, 5)


def test_complete_topology_forward_and_losses_are_finite():
    torch.manual_seed(7)
    model = TopologyEnhancedAssociation(appearance_dim=32, hidden_dim=24, patterns=5, dropout=0.0)
    geometric = torch.rand(6, 7)
    score, left, right = model(_view(6), _view(7), geometric)
    assert score.shape == (6, 7)
    assert left.memberships.shape == (6, 5)
    assert torch.all(left.memberships.sum(1) > 0)
    assert torch.all(left.memberships.sum(1) <= 1.0 + 1e-5)
    pairs = torch.tensor([[0, 1], [2, 3], [4, 5]])
    loss, parts = supervised_alignment_loss(left, right, pairs)
    total = loss + membership_regularization(left) + membership_regularization(right)
    assert torch.isfinite(total)
    assert set(parts) == {"identity", "context"}
    total.backward()
    assert model.topology.patterns.grad is not None


def test_purification_preserves_absolute_reliability_strength():
    for purification in ("none", "hard", "continuous"):
        model = TopologyEnhancedAssociation(
            appearance_dim=32, hidden_dim=16, patterns=4, dropout=0.0,
            purification=purification, hard_threshold=0.0,
        )
        _, left, _ = model(_view(3), _view(4), torch.rand(3, 4))
        sums = left.memberships.sum(1)
        assert torch.all(sums > 0)
        assert torch.all(sums <= 1.0 + 1e-5)
        if purification == "none":
            assert torch.allclose(sums, torch.ones(3), atol=1e-5)


def test_true_feature_deletion_and_purification_floor():
    for mode, missing in (("h_only", "context_head"), ("g_only", "identity_head"), ("h_plus_g", None)):
        model = TopologyEnhancedAssociation(
            appearance_dim=32, hidden_dim=16, patterns=4, dropout=0.0,
            feature_mode=mode, purification_floor=0.0,
        )
        if missing:
            assert getattr(model.topology, missing) is None
        score, left, _ = model(_view(3), _view(4), torch.rand(3, 4))
        assert torch.isfinite(score).all()
        assert torch.all(left.memberships.sum(1) <= 1.0 + 1e-5)


def test_all_topology_ablation_modes_run():
    for topology_mode in ("knn_graph", "graph_learning", "fixed_hypergraph", "dynamic_hypergraph"):
        model = TopologyEnhancedAssociation(
            appearance_dim=32, hidden_dim=16, patterns=4, dropout=0.0,
            topology_mode=topology_mode,
        )
        score, left, right = model(_view(5), _view(6), torch.rand(5, 6))
        assert score.shape == (5, 6)
        assert left.context.shape == (5, 16)
        assert right.identity.shape == (6, 16)


def test_batched_path_matches_single_window_path_in_eval_mode():
    torch.manual_seed(9)
    for topology_mode in ("knn_graph", "graph_learning", "fixed_hypergraph", "dynamic_hypergraph"):
        model = TopologyEnhancedAssociation(
            appearance_dim=8, hidden_dim=16, patterns=4, dropout=0.0,
            topology_mode=topology_mode,
        ).eval()
        left = (torch.randn(5, 8), torch.randn(5, 4), torch.randn(5, 5))
        right = (torch.randn(6, 8), torch.randn(6, 4), torch.randn(6, 5))
        geometric = torch.randn(5, 6)
        single, _, _ = model(left, right, geometric)
        batched, _, _ = model.forward_batched(
            tuple(value[None] for value in left), tuple(value[None] for value in right),
            torch.ones(1, 5, dtype=torch.bool), torch.ones(1, 6, dtype=torch.bool),
            geometric[None],
        )
        assert torch.allclose(single, batched[0], atol=1e-5, rtol=1e-4), topology_mode


def test_single_view_observation_can_update_causal_encoder_without_candidates():
    model = TopologyEnhancedAssociation(
        appearance_dim=8, hidden_dim=16, patterns=4, dropout=0.0,
    ).eval()
    empty = (torch.empty(0, 8), torch.empty(0, 4), torch.empty(0, 5))
    score, left, right = model(_view(3, 8), empty, torch.empty(3, 0))
    assert score.shape == (3, 0)
    assert left.semantic.shape == (3, 32)
    assert right.semantic.shape == (0, 32)


def test_candidate_competition_margin_grows_with_candidate_count():
    scores = torch.tensor([[0.90, 0.80, 0.10], [0.20, 0.85, 0.10]]).numpy()
    assert len(competition_assignment(scores, base_margin=0.05, candidate_growth=0.0)) == 2
    assert len(competition_assignment(scores, base_margin=0.05, candidate_growth=0.20)) < 2


def test_alpha_is_geometry_weight_from_paper_fusion():
    model = TopologyEnhancedAssociation(
        appearance_dim=8, hidden_dim=16, patterns=4, dropout=0.0, alpha=1.0,
    ).eval()
    geometric = torch.rand(3, 4)
    score, _, _ = model(_view(3, 8), _view(4, 8), geometric)
    assert torch.allclose(score, geometric)


def test_relation_semantic_is_parameter_free_purified_pattern_sum():
    torch.manual_seed(12)
    model = TopologyEnhancedAssociation(
        appearance_dim=8, hidden_dim=16, patterns=4, dropout=0.0,
    ).eval()
    _, output, _ = model(_view(3, 8), _view(2, 8), torch.rand(3, 2))
    expected = torch.nn.functional.normalize(output.memberships @ model.topology.patterns, dim=-1)
    assert torch.allclose(output.context, expected, atol=1e-6)


def test_node_representation_adds_three_independent_d_dimensional_projections():
    torch.manual_seed(21)
    model = TopologyEnhancedAssociation(
        appearance_dim=8, hidden_dim=16, patterns=4, dropout=0.0,
    ).eval()
    appearance, spatial, trajectory = _view(3, 8)
    _, output, _ = model((appearance, spatial, trajectory), _view(2, 8), torch.rand(3, 2))
    expected = torch.nn.functional.normalize(
        model.topology.appearance_projection(appearance)
        + model.topology.spatial_projection(spatial)
        + model.topology.trajectory_projection(trajectory), dim=-1,
    )
    assert output.nodes.shape == (3, 16)
    assert torch.allclose(output.nodes, expected, atol=1e-6)


def test_causal_history_aggregates_h_only_and_keeps_current_g():
    model = TopologyEnhancedAssociation(
        appearance_dim=8, hidden_dim=16, patterns=4, dropout=0.0,
    ).eval()
    _, output, _ = model(_view(1, 8), _view(1, 8), torch.rand(1, 1))
    key = ("26", 1, 9, 4, 0)
    memory = defaultdict(lambda: deque(maxlen=2))
    memory[("26", 1, 4, 0)].append(-output.identity[0])
    semantic = temporal_semantics(output, [key], memory, 0.25, 2, "h_plus_g")
    assert torch.allclose(
        torch.nn.functional.normalize(semantic[0, 16:], dim=0), output.context[0], atol=1e-6
    )
    expected_h = torch.nn.functional.normalize(0.75 * output.identity[0] - 0.25 * output.identity[0], dim=0)
    assert torch.allclose(
        torch.nn.functional.normalize(semantic[0, :16], dim=0), expected_h, atol=1e-6
    )


def test_competition_never_selects_geometry_masked_candidate():
    scores = torch.tensor([[0.95, 0.70], [0.80, 0.60]]).numpy()
    mask = torch.tensor([[False, True], [True, True]]).numpy()
    selected = competition_assignment(scores, mask, base_margin=0.0, candidate_growth=0.0)
    assert all((row, col) != (0, 0) for row, col, _ in selected)


def test_competition_uses_independent_reverse_direction_scores():
    forward = torch.tensor([[0.9, 0.1], [0.2, 0.8]]).numpy()
    reverse = torch.tensor([[0.1, 0.9], [0.8, 0.2]]).numpy()
    selected = competition_assignment(forward, reverse_scores=reverse)
    assert selected == []


def test_competition_singleton_can_choose_no_match_by_absolute_fused_score():
    # A singleton has no runner-up, so its relative margin is infinite.  The
    # final fused-score gate must still be able to reject a weak ID merge.
    scores = torch.tensor([[0.31]]).numpy()
    assert competition_assignment(scores, minimum_score=0.50) == []
    assert len(competition_assignment(scores, minimum_score=0.30)) == 1


def test_repair_positive_weight_focuses_loss_on_hard_mismatched_pair():
    scores = torch.tensor([[2.0, 0.0], [2.0, 0.0]])
    pairs = torch.tensor([[0, 0], [1, 1]])
    plain = association_loss(scores, pairs, temperature=1.0)
    focused = association_loss(
        scores, pairs, temperature=1.0, positive_weights=torch.tensor([1.0, 8.0])
    )
    assert focused > plain
