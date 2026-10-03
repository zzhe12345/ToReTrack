from pathlib import Path
import torch

from uav_tracking.cache_types import Observation, WindowTrack
from uav_tracking.identity_topology_data import PredictedTopologyFeatureDataset, trajectory_features


def test_trajectory_features_have_documented_shapes():
    records = [
        (0, {"bbox": [0, 0, 20, 10]}),
        (1, {"bbox": [2, 1, 22, 11]}),
        (2, {"bbox": [4, 2, 24, 12]}),
    ]
    spatial, trajectory = trajectory_features(records, 0, 4, 100, 50)
    assert spatial.shape == (4,)
    assert trajectory.shape == (5,)
    assert torch.allclose(spatial, torch.tensor([0.14, 0.14, 0.20, 0.20]))
    assert torch.allclose(trajectory, torch.tensor([0.02, 0.02, 0.0, 0.0, 1.0]))


def test_trajectory_uses_one_deterministic_state_per_frame():
    records = [
        (0, {"bbox": [0, 0, 10, 10]}),
        (1, {"bbox": [1, 0, 11, 10]}),
        (1, {"bbox": [50, 50, 60, 60]}),
    ]
    spatial, trajectory = trajectory_features(records, 0, 2, 100, 100)
    assert torch.allclose(spatial, torch.tensor([0.06, 0.05, 0.10, 0.10]))
    assert torch.allclose(trajectory, torch.tensor([0.01, 0.0, 0.0, 0.0, 1.0]))


def test_predicted_dataset_uses_tracklet_observations_and_gt_only_for_pairs():
    items = []
    truth = {}
    for view in (1, 2):
        for track, gt_id in ((10, 1), (20, 2)):
            index = len(items)
            items.append(WindowTrack("26", view, track, 0, 4, [
                Observation(frame, [track, frame, track + 5, frame + 4], Path("unused.jpg"), 20.0)
                for frame in range(4)
            ]))
            truth[index] = {"gt_id": gt_id, "purity": 1.0, "tracking_quality": 1.0}
    fixture_dir = Path("outputs/test_identity_topology_fixture")
    fixture_dir.mkdir(parents=True, exist_ok=True)
    prediction = fixture_dir / "prediction.pt"
    torch.save({"items": items, "truth": truth, "args": {"xml": str(fixture_dir / "missing_xml")}}, prediction)
    cache = fixture_dir / "cache.pt"
    torch.save({
        "metadata": {"item_keys": [item.key for item in items]},
        "prototypes": {"mean": torch.eye(4), "robust_mean": torch.eye(4)},
        "mean_qualities": torch.ones(4),
    }, cache)
    dataset = PredictedTopologyFeatureDataset(cache, prediction, image_size=(100, 100))
    sample = dataset[0]
    assert len(dataset) == 1
    assert sample["positive_pairs"].tolist() == [[0, 0], [1, 1]]
    assert sample["left"][1].shape == (2, 4)
