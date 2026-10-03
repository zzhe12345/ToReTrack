"""Test trained weights using the config written by train.py."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from run_pipeline import ROOT, resolve_path, run_step, runtime_environment


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, help="Defaults to <work_root>/trained_pipeline.json")
    parser.add_argument("--scenes", default="", help="Empty tests the complete official test split")
    parser.add_argument("--max-frames", type=int, default=0, help="Engineering check only")
    parser.add_argument("--visualize", action="store_true")
    args = parser.parse_args()
    base = json.loads((ROOT / "configs/pipeline.json").read_text(encoding="utf-8"))
    config_path = args.config or (resolve_path(base["work_root"]) / "trained_pipeline.json")
    if not config_path.is_file():
        parser.error(f"Training config not found: {config_path}. Run train.py first or specify --config.")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    env = runtime_environment(resolve_path(config["weights_root"]))
    arguments = ["--config", str(config_path.resolve()), "--split", "test",
                 "--scenes", args.scenes, "--max-frames", str(args.max_frames)]
    run_step("run_pipeline.py", ["--phase", "infer", *arguments,
                                *(["--visualize"] if args.visualize else [])], env)
    run_step("run_pipeline.py", ["--phase", "evaluate", *arguments], env)


if __name__ == "__main__":
    main()
