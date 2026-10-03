"""Evaluate final P2 results: MDA plus per-view and pooled MOT metrics."""
from __future__ import annotations

import argparse
import contextlib
import importlib.util
import io
import json
from pathlib import Path

import motmetrics as mm
import numpy as np
import pandas as pd

# motmetrics 1.4 still calls this NumPy 1.x alias.
if not hasattr(np, "asfarray"):
    np.asfarray = lambda value: np.asarray(value, dtype=float)  # type: ignore[attr-defined]


SCENES = (26, 31, 34, 48, 52, 55, 56, 57, 59, 61, 62, 68, 71, 73)


def load_mot_gt(path: Path) -> pd.DataFrame:
    return mm.io.loadtxt(str(path), fmt="mot15-2D", min_confidence=1)


def json_to_mot(path: Path) -> pd.DataFrame:
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = []
    for frame0 in range(len(data)):
        for item in data.get(f"frame={frame0}", []):
            track_id, x1, y1, x2, y2 = item[:5]
            # The released MDMT MOT text files use zero-based frame IDs.
            # mm.io.loadtxt applies the MOTChallenge 1-based-to-0-based spatial
            # conversion to both GT and submitted TXT files.  JSON predictions
            # bypass that loader, so apply the same conversion here; otherwise
            # every prediction is displaced by (+1, +1) relative to loaded GT.
            rows.append((frame0, int(track_id), x1 - 1, y1 - 1, x2 - x1, y2 - y1, 1.0, 1, 1, -1))
    frame = pd.DataFrame(rows, columns=["FrameId", "Id", "X", "Y", "Width", "Height", "Confidence", "ClassId", "Visibility", "unused"])
    if frame.empty:
        return frame.set_index(["FrameId", "Id"])
    return frame.set_index(["FrameId", "Id"]).sort_index()


def released_mango_mda(results: Path, gt: Path, scenes) -> float:
    """Invoke the released ``demo/eval/mango_eval.py`` implementation verbatim.

    The released evaluator expects the one-based files under ``demo/eval/test``
    and subtracts one internally; these are not byte-identical to the
    zero-based MOT-evaluation copy under ``demo/txt/gt_true``.
    """
    source = Path(__file__).resolve().parent / "third_party/mia_net_official/demo/eval/mango_eval.py"
    spec = importlib.util.spec_from_file_location("mia_released_mango_eval", source)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import released evaluator: {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    per_scene = {}
    for scene in scenes:
        frames = module.read_result(results / f"{scene}-1.json", results / f"{scene}-2.json")
        gt_a = module.read_gt(gt / f"{scene}-1.txt", 1)
        gt_b = module.read_gt(gt / f"{scene}-2.txt", 2)
        frames = module.assign_gt_to_frames(frames, gt_a, gt_b)
        with contextlib.redirect_stdout(io.StringIO()):
            per_scene[str(scene)] = float(module.calAAS(frames))
    return float(np.mean(list(per_scene.values())))


def summarize_mot(accumulators: list, names: list[str]) -> dict:
    """Pool independent sequences within each view and across both views.

    Track IDs remain local to each accumulator. MOTA/IDF1 are recomputed from
    accumulated counts, rather than added or averaged across sequences.
    IDS is a count and its total is the sum of the two views.
    """
    if not accumulators or len(accumulators) != len(names):
        raise ValueError("Provide matching nonempty accumulators and sequence names")
    handler = mm.metrics.create()
    output = {}
    for label, view in [("view_1", 1), ("view_2", 2), ("total", None)]:
        indices = [i for i, name in enumerate(names)
                   if view is None or name.endswith(f"-{view}")]
        if not indices:
            raise ValueError(f"No sequences supplied for {label}")
        summary = handler.compute_many(
            [accumulators[i] for i in indices], names=[names[i] for i in indices],
            metrics=["mota", "idf1", "num_switches"], generate_overall=True,
        )
        row = summary.loc["OVERALL"]
        output[label] = {"MOTA": 100 * float(row.mota),
                         "IDF1": 100 * float(row.idf1), "IDS": int(row.num_switches)}
    return output


def print_final_metrics(result: dict) -> None:
    print(f"MDA: {result['MDA']:.2f}%")
    print(f"{'Scope':<10} {'MOTA (%)':>12} {'IDF1 (%)':>12} {'IDS':>10}")
    for key, label in [("view_1", "View 1"), ("view_2", "View 2"), ("total", "Total")]:
        row = result[key]
        print(f"{label:<10} {row['MOTA']:>12.2f} {row['IDF1']:>12.2f} {row['IDS']:>10d}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--work", required=True, type=Path,
                    help="Test work directory; always reads final p2/ and mot_gt/, mda_gt/.")
    ap.add_argument("--scenes", default="", help="Comma-separated subset; empty uses the 14 test scenes.")
    args = ap.parse_args()
    scenes = tuple(int(value) for value in args.scenes.split(",") if value.strip()) or SCENES
    if len(set(scenes)) != len(scenes) or any(scene not in SCENES for scene in scenes):
        ap.error("Choose distinct scenes from the official test split")
    results = args.work / "p2"

    accs, names = [], []
    for scene in scenes:
        for view in (1, 2):
            name = f"{scene}-{view}"
            gt = load_mot_gt(args.work / "mot_gt" / f"{name}.txt")
            pred = json_to_mot(results / f"{name}.json")
            accs.append(mm.utils.compare_to_groundtruth(gt, pred, "iou", distth=0.5))
            names.append(name)
    result = {"MDA": 100 * released_mango_mda(results, args.work / "mda_gt", scenes),
              **summarize_mot(accs, names)}
    print_final_metrics(result)
    output = args.work / "metrics.json"
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
