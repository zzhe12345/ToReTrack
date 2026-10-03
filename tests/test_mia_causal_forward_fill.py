from collections import defaultdict, deque

import numpy as np

from apply_mia_causal_forward_fill import fill_scene, predict_box


def test_prediction_uses_only_past_observations():
    history = deque([
        (0, np.asarray([0, 0, 10, 10], dtype=np.float64)),
        (1, np.asarray([1, 0, 11, 10], dtype=np.float64)),
    ])
    assert np.allclose(predict_box(history, 2, 1.0), [2, 0, 12, 10])


def test_median_velocity_rejects_one_noisy_step():
    history = deque([
        (0, np.asarray([0, 0, 10, 10], dtype=np.float64)),
        (1, np.asarray([1, 0, 11, 10], dtype=np.float64)),
        (2, np.asarray([20, 0, 30, 10], dtype=np.float64)),
        (3, np.asarray([21, 0, 31, 10], dtype=np.float64)),
    ])
    predicted = predict_box(history, 4, 1.0, "median")
    assert np.allclose(predicted, [22, 0, 32, 10])


def test_other_view_guard_inserts_one_frame_prediction():
    left = {
        "frame=0": [[7, 0, 0, 10, 10]],
        "frame=1": [[7, 1, 0, 11, 10]],
        "frame=2": [],
    }
    right = {
        "frame=0": [[7, 100, 0, 110, 10]],
        "frame=1": [[7, 101, 0, 111, 10]],
        "frame=2": [[7, 102, 0, 112, 10]],
    }
    report = fill_scene(
        [left, right], maximum_age=1, maximum_overlap=0.1,
        velocity_decay=1.0, minimum_observations=2,
        require_other_view=True, width=1920, height=1080,
        minimum_visible_fraction=0.5,
    )
    assert left["frame=2"] == [[7, 2.0, 0.0, 12.0, 10.0]]
    assert report["1"]["inserted"] == 1


def test_future_return_does_not_rewrite_previous_prediction():
    left = {
        "frame=0": [[7, 0, 0, 10, 10]],
        "frame=1": [[7, 1, 0, 11, 10]],
        "frame=2": [],
        "frame=3": [[7, 30, 0, 40, 10]],
    }
    right = {f"frame={frame}": [[7, 100 + frame, 0, 110 + frame, 10]] for frame in range(4)}
    fill_scene(
        [left, right], maximum_age=1, maximum_overlap=0.1,
        velocity_decay=1.0, minimum_observations=2,
        require_other_view=True, width=1920, height=1080,
        minimum_visible_fraction=0.5,
    )
    assert left["frame=2"][0][1:5] == [2.0, 0.0, 12.0, 10.0]


def test_current_other_view_geometry_can_correct_prediction():
    left = {
        "frame=0": [[7, 0, 0, 10, 10]],
        "frame=1": [[7, 0, 0, 10, 10]],
        "frame=2": [],
    }
    right = {
        "frame=0": [[7, 10, 0, 20, 10]],
        "frame=1": [[7, 11, 0, 21, 10]],
        "frame=2": [[7, 12, 0, 22, 10]],
    }
    traces = {"frame=2": {
        "view2_to_view1": [[1, 0, -10], [0, 1, 0], [0, 0, 1]],
        "view1_to_view2": [[1, 0, 10], [0, 1, 0], [0, 0, 1]],
    }}
    fill_scene(
        [left, right], maximum_age=1, maximum_overlap=0.1,
        velocity_decay=0.0, minimum_observations=2,
        require_other_view=True, width=1920, height=1080,
        minimum_visible_fraction=0.5, geometry_traces=traces,
        cross_view_weight=1.0,
    )
    assert np.allclose(left["frame=2"][0][1:5], [2, 0, 12, 10])
