"""Check data conversion and validation-only repair threshold selection."""
import json
from PIL import Image

from prepare_detection_data import convert
from train import choose_threshold


def test_conversion_keeps_autoassign_classes_and_esod_proposals(tmp_path):
    data = tmp_path / "data"
    for split, scene in [("train", 23), ("val", 22)]:
        for view in (1, 2):
            frames = data / split / str(view) / f"{scene}-{view}"
            labels = data / "new_xml" / str(view)
            frames.mkdir(parents=True)
            labels.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (100, 80)).save(frames / "000000.jpg")
            (labels / f"{scene}-{view}.xml").write_text(
                '<annotations><track id="0" label="car">'
                '<box frame="0" outside="0" xtl="10" ytl="20" xbr="30" ybr="40"/>'
                '</track></annotations>', encoding="utf-8")
    output = tmp_path / "prepared"
    convert(data, output)
    coco = json.loads((output / "train_coco.json").read_text())
    assert len(coco["images"]) == 2
    assert [row["category_id"] for row in coco["annotations"]] == [3, 3]
    assert coco["annotations"][0]["bbox"] == [10, 20, 20, 20]
    yolo = (output / "labels/train/23_1_000000.txt").read_text().split()
    assert yolo == ["0", "0.20000000", "0.37500000", "0.20000000", "0.25000000"]
    assert all(path.is_file() for path in output.glob("*_coco.json"))


def test_threshold_selection_rejects_high_recall_unreliable_repairs():
    rows = [
        {"threshold": .2, "repair_precision": .5, "repair_recall": .9, "repair_f0_5": .6, "mda": .9, "idf1": .9},
        {"threshold": .8, "repair_precision": .9, "repair_recall": .3, "repair_f0_5": .6, "mda": .8, "idf1": .8},
    ]
    assert choose_threshold({"threshold_sweep": rows}) == .8
    assert choose_threshold({"threshold_sweep": []}) > 1
