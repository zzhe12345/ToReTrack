"""Verify final-only evaluation with unequal sequence lengths and an ID switch."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import evaluate_mia_metrics as evaluator
import run_pipeline


class FinalMetricsTests(unittest.TestCase):
    def make_work(self, work: Path) -> None:
        for name in ("p0", "p1", "p2", "mot_gt", "mda_gt"):
            (work / name).mkdir()
        for scene, frames in [(26, 2), (31, 8)]:
            for view in (1, 2):
                sequence = f"{scene}-{view}"
                for offset, folder in [(0, "mot_gt"), (1, "mda_gt")]:
                    lines = [f"{frame + offset},1,20,20,10,10,1,1,1,-1\n"
                             for frame in range(frames)]
                    (work / folder / f"{sequence}.txt").write_text("".join(lines))
                predictions = {}
                for frame in range(frames):
                    identity = 11 if scene == 26 and view == 1 and frame == 1 else 10
                    missing = scene == 31 and view == 2 and frame >= 4
                    predictions[f"frame={frame}"] = [] if missing else [[identity, 20, 20, 30, 30]]
                (work / "p2" / f"{sequence}.json").write_text(json.dumps(predictions))
                # Earlier-stage files must never be opened by the final evaluator.
                for folder in ("p0", "p1"):
                    (work / folder / f"{sequence}.json").write_text("invalid intermediate result")

    def test_final_report_pools_counts_and_ignores_intermediate_results(self):
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            self.make_work(work)
            terminal = io.StringIO()
            with patch("sys.argv", ["evaluate_mia_metrics.py", "--work", str(work), "--scenes", "26,31"]):
                with contextlib.redirect_stdout(terminal):
                    evaluator.main()
            result = json.loads((work / "metrics.json").read_text())
            self.assertEqual(set(result), {"MDA", "view_1", "view_2", "total"})
            self.assertAlmostEqual(result["MDA"], 50)
            expected = {"view_1": (90, 90, 1), "view_2": (60, 75, 0),
                        "total": (75, 100 * 30 / 36, 1)}
            for key, (mota, idf1, ids) in expected.items():
                self.assertEqual(set(result[key]), {"MOTA", "IDF1", "IDS"})
                self.assertAlmostEqual(result[key]["MOTA"], mota)
                self.assertAlmostEqual(result[key]["IDF1"], idf1)
                self.assertEqual(result[key]["IDS"], ids)
            self.assertNotAlmostEqual(result["total"]["IDF1"], (90 + 75) / 2)
            for label in ("MDA:", "View 1", "View 2", "Total"):
                self.assertIn(label, terminal.getvalue())
            self.assertNotIn("per_sequence", terminal.getvalue())

    def test_arbitrary_results_argument_is_not_provided(self):
        with patch("sys.argv", ["evaluate_mia_metrics.py", "--work", "unused", "--results", "p0"]):
            with contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as raised:
                    evaluator.main()
        self.assertEqual(raised.exception.code, 2)

    def test_pipeline_uses_final_evaluator_without_stage_selection(self):
        class Arguments:
            split = "test"
            max_frames = 0
        with patch.object(run_pipeline, "run_step") as step:
            run_pipeline.evaluate(Arguments(), {}, Path("external"), Path("work/test"), "26")
        calls = step.call_args_list
        self.assertEqual(len(calls), 3)
        self.assertEqual(calls[-1].args[0], "evaluate_mia_metrics.py")
        self.assertEqual(calls[-1].args[1], ["--work", str(Path("work/test")), "--scenes", "26"])


if __name__ == "__main__":
    unittest.main()
