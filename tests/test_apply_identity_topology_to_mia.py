import json
from pathlib import Path
from types import SimpleNamespace

from apply_identity_topology_to_mia import apply_synchronous_online


def test_synchronous_noop_preserves_official_mia_ids():
    fixture_root = Path("outputs/test_identity_topology_fixture/noop_fallback")
    mia_results = fixture_root / "mia"
    output = fixture_root / "output"
    mia_results.mkdir(parents=True, exist_ok=True)
    view1 = {
        "frame=0": [[7, 10.0, 20.0, 30.0, 40.0, 0.9]],
        "frame=1": [[7, 11.0, 21.0, 30.0, 40.0, 0.8]],
    }
    view2 = {
        "frame=0": [[7, 50.0, 60.0, 20.0, 25.0, 0.7]],
        "frame=1": [[9, 51.0, 61.0, 20.0, 25.0, 0.6]],
    }
    (mia_results / "1-1.json").write_text(json.dumps(view1), encoding="utf-8")
    (mia_results / "1-2.json").write_text(json.dumps(view2), encoding="utf-8")
    dataset = SimpleNamespace(samples=[
        {
            "left_keys": [("1", 1, 0, 7, 0)],
            "right_keys": [("1", 2, 0, 7, 0)],
        },
        {
            "left_keys": [("1", 1, 1, 7, 0)],
            "right_keys": [("1", 2, 1, 9, 0)],
        },
    ])
    records = [
        {"scene": "1", "frame": 0, "pairs": []},
        {"scene": "1", "frame": 1, "pairs": []},
    ]

    report = apply_synchronous_online(records, dataset, mia_results, output)

    assert json.loads((output / "1-1.json").read_text(encoding="utf-8")) == view1
    assert json.loads((output / "1-2.json").read_text(encoding="utf-8")) == view2
    assert report["accepted_current_frame_links"] == 0
    assert report["topology_identity_corrections"] == 0


def test_synchronous_correction_rejects_within_view_identity_collision():
    fixture_root = Path("outputs/test_identity_topology_fixture/collision_guard")
    mia_results = fixture_root / "mia"
    output = fixture_root / "output"
    mia_results.mkdir(parents=True, exist_ok=True)
    view1 = {"frame=0": [
        [7, 10.0, 20.0, 30.0, 40.0, 0.9],
        [9, 60.0, 20.0, 30.0, 40.0, 0.8],
    ]}
    view2 = {"frame=0": [
        [7, 10.0, 60.0, 30.0, 40.0, 0.9],
        [9, 60.0, 60.0, 30.0, 40.0, 0.8],
    ]}
    (mia_results / "1-1.json").write_text(json.dumps(view1), encoding="utf-8")
    (mia_results / "1-2.json").write_text(json.dumps(view2), encoding="utf-8")
    dataset = SimpleNamespace(samples=[{
        "left_keys": [("1", 1, 0, 7, 0), ("1", 1, 0, 9, 0)],
        "right_keys": [("1", 2, 0, 7, 0), ("1", 2, 0, 9, 0)],
    }])
    records = [{"scene": "1", "frame": 0, "pairs": [(0, 1, 1.0)]}]

    report = apply_synchronous_online(records, dataset, mia_results, output)

    assert json.loads((output / "1-1.json").read_text(encoding="utf-8")) == view1
    assert json.loads((output / "1-2.json").read_text(encoding="utf-8")) == view2
    assert report["topology_identity_corrections"] == 0
    assert report["collision_rejected_corrections"] == 1


def test_synchronous_correction_applies_when_temporal_supports_do_not_overlap():
    fixture_root = Path("outputs/test_identity_topology_fixture/nonoverlap_merge")
    mia_results = fixture_root / "mia"
    output = fixture_root / "output"
    mia_results.mkdir(parents=True, exist_ok=True)
    view1 = {"frame=0": [[7, 10.0, 20.0, 30.0, 40.0, 0.9]]}
    view2 = {"frame=0": [[9, 50.0, 60.0, 20.0, 25.0, 0.7]]}
    (mia_results / "1-1.json").write_text(json.dumps(view1), encoding="utf-8")
    (mia_results / "1-2.json").write_text(json.dumps(view2), encoding="utf-8")
    dataset = SimpleNamespace(samples=[{
        "left_keys": [("1", 1, 0, 7, 0)],
        "right_keys": [("1", 2, 0, 9, 0)],
    }])
    records = [{"scene": "1", "frame": 0, "pairs": [(0, 0, 1.0)]}]

    report = apply_synchronous_online(records, dataset, mia_results, output)

    result = json.loads((output / "1-2.json").read_text(encoding="utf-8"))
    assert result["frame=0"][0][0] == 7
    assert report["topology_identity_corrections"] == 1
    assert report["collision_rejected_corrections"] == 0
