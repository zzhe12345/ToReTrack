"""Run detection, cross-view association, topology training and P2 inference.

All datasets and model weights are supplied by the user. Generated files stay
under the configured work directory; no precomputed experiment is required.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent


def resolve_path(value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def runtime_environment(weight_root: Path) -> dict[str, str]:
    env = os.environ.copy()
    # Keep library temporary files local, including Windows/YAPF caches.
    runtime_root = ROOT / "tmp/runtime"
    runtime_root.mkdir(parents=True, exist_ok=True)
    env["TMP"] = env["TEMP"] = str(runtime_root)
    env["MPLCONFIGDIR"] = str(runtime_root)
    env["MIA_YAPF_CACHE"] = str(runtime_root / "yapf")
    env["PYTHONPATH"] = os.pathsep.join([
        str(ROOT / "src"), str(ROOT / "third_party/mia_net_official"),
        env.get("PYTHONPATH", ""),
    ])
    env["UAV_TRACKING_WEIGHT_ROOT"] = str(weight_root)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return env


def run_step(script: str, arguments: list[str], env: dict[str, str]) -> None:
    # Always rerun requested stages: existing output alone cannot prove that
    # it was generated with the current inputs, weights or parameters.
    print(f"\nRunning {script}", flush=True)
    subprocess.run([sys.executable, "-B", str(ROOT / script), *arguments],
                   cwd=ROOT, env=env, check=True)


def check_weights(root: Path, names: list[str]) -> None:
    missing = [str(root / name) for name in names if not (root / name).is_file()]
    if missing:
        raise FileNotFoundError("Provide external model weights:\n" + "\n".join(missing))


def prepare(config: dict, args, env: dict[str, str], data: Path,
            weights: Path, work: Path, scenes: str, *, with_truth: bool) -> None:
    check_weights(weights, ["esod_epoch5.pt", "autoassign_epoch60.pth"])
    if not (data / args.split).is_dir() or not (data / "new_xml").is_dir():
        raise FileNotFoundError(f"Expected {data / args.split} and {data / 'new_xml'}")
    split_file = str(ROOT / "configs/splits.json")
    detections, p0, geometry = work / "detections", work / "p0", work / "geometry"
    common = ["--device", args.device]
    frame_limit = ["--max-frames", str(args.max_frames)] if args.max_frames else []
    run_step("generate_esod_detection_cache.py", [
        "--checkpoint", str(weights / "esod_epoch5.pt"), "--out", str(detections),
        "--split", args.split, "--data", str(data / args.split),
        "--xml", str(data / "new_xml"), "--split-file", split_file,
        "--scene-ids", scenes, "--img-size", "1280", "--batch-size", "2",
        "--conf-thres", "0.01", "--iou-thres", "0.5", *common,
        *(["--max-frames-per-view", str(args.max_frames)] if args.max_frames else []),
    ], env)
    run_step("run_mia_official_geometry.py", [
        "--frontend", "autoassign_bytetrack", "--data", str(data / args.split),
        "--xml-dir", str(data / "new_xml"), "--out", str(p0),
        "--geometry-trace-dir", str(geometry), "--scenes", scenes,
        "--external-detection-cache", str(detections),
        "--external-detection-threshold", "0.10", "--external-replace",
        *common, *frame_limit,
    ], env)
    # Ground truth provides positive pairs only for train/validation caches.
    # Test inference uses frozen parameters and does not consume truth labels.
    run_step("build_mia_frame_topology_cache.py", [
        "--mia-results", str(p0), "--geometry-trace", str(geometry),
        "--out", str(work / "frame_topology_cache.pt"),
        "--frontend", "autoassign_bytetrack", "--split", args.split,
        "--data", str(data), "--xml", str(data / "new_xml"),
        "--split-file", split_file, "--scenes", scenes,
        "--insertion-point", "post_mia_output",
        "--pooling", "assigned_fpn_roi_align_3x3_meanmax",
        *([] if with_truth else ["--without-truth"]), *common, *frame_limit,
    ], env)


def infer(config: dict, args, env: dict[str, str], weights: Path,
          work: Path, scenes: str) -> None:
    check_weights(weights, [config["topology_checkpoint"]])
    policy = config["online_policy"]
    arguments = [
        "--checkpoint", str(weights / config["topology_checkpoint"]),
        "--cache", str(work / "frame_topology_cache.pt"),
        "--mia-results", str(work / "p0"), "--out", str(work / "p1"),
        "--device", args.device,
    ]
    for key in ["base_margin", "candidate_growth", "minimum_score", "candidate_topk",
                "minimum_pair_support", "maximum_pair_gap"]:
        arguments.extend(["--" + key.replace("_", "-"), str(policy[key])])
    if policy["causal_collision_guard"]:
        arguments.append("--causal-collision-guard")
    run_step("apply_identity_topology_to_mia.py", arguments, env)
    recovery = config["recovery"]
    arguments = ["--results", str(work / "p1"), "--out", str(work / "p2"),
                 "--scenes", scenes]
    # This age-5 median-velocity policy is the final P2 configuration.
    # Recovery uses current/past observations only; it never edits past frames.
    for key in ["maximum_age", "maximum_overlap", "velocity_decay",
                "velocity_estimator", "minimum_observations"]:
        arguments.extend(["--" + key.replace("_", "-"), str(recovery[key])])
    if recovery["require_other_view"]:
        arguments.append("--require-other-view")
    run_step("apply_mia_causal_forward_fill.py", arguments, env)


def train(config: dict, args, env: dict[str, str], work_root: Path) -> None:
    arguments = [
        "--train-cache", str(work_root / "train/frame_topology_cache.pt"),
        "--val-cache", str(work_root / "val/frame_topology_cache.pt"),
        "--out", str(work_root / f"models/seed_{args.seed}"),
        "--seed", str(args.seed), "--device", args.device,
    ]
    for key, value in config["training"].items():
        arguments.extend(["--" + key.replace("_", "-"), str(value)])
    run_step("train_identity_topology.py", arguments, env)


def evaluate(args, env: dict[str, str], data: Path, work: Path, scenes: str) -> None:
    for offset, name in [(0, "mot_gt"), (1, "mda_gt")]:
        run_step("build_mot_gt_from_xml.py", [
            "--split", args.split, "--data", str(data), "--xml", str(data / "new_xml"),
            "--split-file", str(ROOT / "configs/splits.json"), "--scenes", scenes,
            "--frame-offset", str(offset), "--max-frames", str(args.max_frames),
            "--out", str(work / name),
        ], env)
    run_step("evaluate_mia_metrics.py", [
        "--work", str(work), "--scenes", scenes,
    ], env)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=["prepare", "infer", "train", "evaluate"], default="infer")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/pipeline.json")
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--scenes", default="", help="Comma-separated IDs; empty processes the split")
    parser.add_argument("--max-frames", type=int, default=0, help="Smoke test only; 0 processes all frames")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", help="Override configured device")
    parser.add_argument("--visualize", action="store_true", help="Render predictions against external labels")
    args = parser.parse_args()
    if args.phase == "evaluate" and args.split != "test":
        parser.error("Evaluation is provided only for final results on the test split")
    if args.max_frames < 0:
        parser.error("--max-frames must be nonnegative")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    args.device = args.device or config["device"]
    data = resolve_path(os.environ.get("UAV_TRACKING_DATA_ROOT", config["data_root"]))
    weights = resolve_path(os.environ.get("UAV_TRACKING_WEIGHT_ROOT", config["weights_root"]))
    work_root = resolve_path(config["work_root"])
    work = work_root / args.split
    splits = json.loads((ROOT / "configs/splits.json").read_text(encoding="utf-8"))["splits"]
    allowed = {str(value) for value in splits[args.split]}
    selected = [value.strip() for value in args.scenes.split(",") if value.strip()]
    if any(value not in allowed for value in selected):
        parser.error(f"Scenes must belong to {args.split}: {sorted(allowed, key=int)}")
    scenes = ",".join(selected or splits[args.split])
    env = runtime_environment(weights)
    if args.phase == "train":
        train(config, args, env, work_root)
    elif args.phase == "evaluate":
        evaluate(args, env, data, work, scenes)
    else:
        if args.phase == "infer":
            check_weights(weights, [config["topology_checkpoint"]])
        prepare(config, args, env, data, weights, work, scenes,
                with_truth=args.phase == "prepare" and args.split != "test")
        if args.phase == "infer":
            infer(config, args, env, weights, work, scenes)
            if args.visualize:
                sys.path.insert(0, str(ROOT / "src"))
                from visualize_results import render_sequence
                for scene in scenes.split(","):
                    render_sequence(sequence=int(scene), split=args.split,
                                    result_dir=work / "p2", output_dir=work / f"visualization/{scene}",
                                    max_frames=args.max_frames, data_root=data)
    print(f"\nCompleted {args.phase}: {work_root if args.phase == 'train' else work}")


if __name__ == "__main__":
    main()
