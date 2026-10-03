"""Train detectors and identity topology, then write a ready-to-test config."""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import shutil
import sys

from run_pipeline import ROOT, resolve_path, run_step, runtime_environment


def choose_threshold(scan: dict) -> float:
    """Choose a repair threshold on validation, never on the test set."""
    rows = scan["threshold_sweep"]
    for precision_floor in (0.8, 0.7):
        candidates = [row for row in rows if row.get("repair_precision", 0) >= precision_floor
                      and row.get("repair_recall", 0) >= 0.1]
        if candidates:
            return float(max(candidates, key=lambda row: (
                row["repair_recall"], row["repair_precision"], row["repair_f0_5"],
                row["mda"], row["idf1"], row["threshold"],
            ))["threshold"])
    # Tiny or newly trained datasets may offer no reliable repair evidence.
    # Disable corrections instead of selecting a threshold using test labels.
    print("Validation has no reliable repair threshold; retaining MIA IDs.")
    return 1.000001


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/pipeline.json")
    parser.add_argument("--stage", choices=["all", "detectors", "topology"], default="all")
    parser.add_argument("--data-root", help="External MDMT directory")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--initial-esod", default="", help="Optional explicit initialization checkpoint")
    parser.add_argument("--initial-autoassign", default="", help="Optional explicit initialization checkpoint")
    parser.add_argument("--max-frames", type=int, default=0, help="Limit frames per sequence for engineering checks")
    parser.add_argument("--smoke", action="store_true", help="One-epoch, small-image check; not a performance run")
    args = parser.parse_args()
    config = deepcopy(json.loads(args.config.read_text(encoding="utf-8")))
    if args.data_root:
        config["data_root"] = args.data_root
    data = resolve_path(config["data_root"])
    work = resolve_path(config["work_root"])
    weights = resolve_path(config["weights_root"])
    if args.smoke and args.max_frames == 0:
        args.max_frames = 2
    for path in (work, weights):
        path.mkdir(parents=True, exist_ok=True)
    config.update(data_root=str(data), work_root=str(work), weights_root=str(weights))
    if args.smoke:
        config["training"]["epochs"] = 1
    working_config = work / "training_pipeline.json"
    working_config.write_text(json.dumps(config, indent=2), encoding="utf-8")
    env = runtime_environment(weights)
    detection = config["detector_training"]
    if args.stage in ("all", "detectors"):
        prepared = work / "detection_data"
        run_step("prepare_detection_data.py", [
            "--data", str(data), "--out", str(prepared), "--max-frames", str(args.max_frames),
        ], env)
        run_step("train_esod.py", [
            "--data", str(prepared / "esod.yaml"),
            "--cfg", str(ROOT / "configs/uavdt_yolov5m.yaml"),
            "--hyp", str(ROOT / "configs/esod_hyp.yaml"),
            "--weights", str(resolve_path(args.initial_esod)) if args.initial_esod else "",
            "--epochs", "1" if args.smoke else str(detection["esod_epochs"]),
            "--batch-size", "2" if args.smoke else str(detection["batch_size"]),
            "--img-size", "512" if args.smoke else str(detection["image_size"]),
            "--device", config["device"].replace("cuda:", ""), "--workers", "0",
            "--project", str(work / "detectors"), "--name", "esod", "--exist-ok",
            "--single-cls", "--notest", "--noautoanchor", "--disable-half",
        ], env)
        esod_checkpoint = work / "detectors/esod/weights/last.pt"
        if not esod_checkpoint.is_file():
            raise FileNotFoundError(esod_checkpoint)
        shutil.copy2(esod_checkpoint, weights / "esod_epoch5.pt")
        run_step("train_autoassign.py", [
            "--annotations", str(prepared / "train_coco.json"),
            "--out", str(weights / "autoassign_epoch60.pth"),
            "--epochs", "1" if args.smoke else str(detection["autoassign_epochs"]),
            "--batch-size", "1" if args.smoke else str(detection["batch_size"]),
            "--device", config["device"], "--seed", str(args.seed),
            *(["--image-size", "320"] if args.smoke else []),
            *(["--initial-weights", str(resolve_path(args.initial_autoassign))] if args.initial_autoassign else []),
        ], env)
    if args.stage in ("all", "topology"):
        for split in ("train", "val"):
            run_step("run_pipeline.py", [
                "--phase", "prepare", "--config", str(working_config), "--split", split,
                "--max-frames", str(args.max_frames),
            ], env)
        run_step("run_pipeline.py", [
            "--phase", "train", "--config", str(working_config), "--seed", str(args.seed),
        ], env)
        topology = work / f"models/seed_{args.seed}/best_checkpoint.pt"
        if not topology.is_file():
            raise FileNotFoundError(topology)
        exported = weights / f"identity_topology_seed{args.seed}.pt"
        shutil.copy2(topology, exported)
        scan = work / "validation_thresholds.json"
        run_step("evaluate_identity_topology.py", [
            "--checkpoint", str(exported), "--cache", str(work / "val/frame_topology_cache.pt"),
            "--tune-threshold", "--out", str(scan), "--device", config["device"],
        ], env)
        config["topology_checkpoint"] = exported.name
        config["online_policy"]["minimum_score"] = choose_threshold(json.loads(scan.read_text(encoding="utf-8")))
        trained_config = work / "trained_pipeline.json"
        trained_config.write_text(json.dumps(config, indent=2), encoding="utf-8")
        print(f"Training complete. Test with: {sys.executable} test.py --config {trained_config}")


if __name__ == "__main__":
    main()
