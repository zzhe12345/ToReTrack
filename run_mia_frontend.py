from __future__ import annotations

"""Run the downloaded official MIA-Net detector + ByteTrack frontends."""

import argparse
import contextlib
import faulthandler
import json
import os
from pathlib import Path
import tempfile

import numpy as np
import torch

from uav_tracking.mmcv_ops_compat import install as install_mmcv_ops


@contextlib.contextmanager
def fixed_mmcv_config_tempdir():
    """Avoid inaccessible sandbox ACLs on Python TemporaryDirectory folders.

    MMCV 1.x copies every config (including bases) through a temporary folder.
    On the Windows workspace sandbox, ``tempfile.TemporaryDirectory`` receives
    a process-only ACL and its immediately-created child file is inaccessible.
    Reuse one pre-created workspace directory during config parsing instead.
    """
    root = Path(os.environ.get("MIA_MMCV_CONFIG_TMP", "tmp/mia_mmcv_config")).resolve()
    root.mkdir(parents=True, exist_ok=True)

    class FixedTemporaryDirectory:
        def __init__(self, *_, **__):
            self.name = str(root)

        def __enter__(self):
            return self.name

        def __exit__(self, *_):
            return False

    original = tempfile.TemporaryDirectory
    tempfile.TemporaryDirectory = FixedTemporaryDirectory
    try:
        yield
    finally:
        tempfile.TemporaryDirectory = original


def detector_checkpoint(frontend: str) -> Path:
    """Resolve external weights without searching any previous project."""
    weight_root = Path(os.environ.get("UAV_TRACKING_WEIGHT_ROOT", "weights"))
    name = "carafe_epoch12.pth" if frontend == "carafe_bytetrack" else "autoassign_epoch60.pth"
    return weight_root / name


def build(frontend: str, device: str, tracker_overrides: dict | None = None,
          test_scale: tuple[int, int] | None = None):
    print(f"building {frontend}: install compatibility ops", flush=True)
    install_mmcv_ops()
    from mmcv import Config
    from mmcv.runner import load_checkpoint
    from mmtrack.models import build_model

    root = Path("third_party/mia_net_official/configs/mot/bytetrack")
    if frontend == "carafe_bytetrack":
        config_path = root / "one_carafe_bytetrack_full_mdmt.py"
    else:
        config_path = root / "bytetrack_autoassign_full_mdmt-private-half.py"
    checkpoint = detector_checkpoint(frontend)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Provide the detector weight file: {checkpoint}")
    with fixed_mmcv_config_tempdir():
        config = Config.fromfile(str(config_path))
    if tracker_overrides:
        tracker = config.model.tracker
        if tracker_overrides.get("high") is not None:
            tracker.obj_score_thrs.high = float(tracker_overrides["high"])
        if tracker_overrides.get("low") is not None:
            tracker.obj_score_thrs.low = float(tracker_overrides["low"])
        if tracker_overrides.get("init") is not None:
            tracker.init_track_thr = float(tracker_overrides["init"])
        if tracker_overrides.get("match_high") is not None:
            tracker.match_iou_thrs.high = float(tracker_overrides["match_high"])
        if tracker_overrides.get("match_low") is not None:
            tracker.match_iou_thrs.low = float(tracker_overrides["match_low"])
        if tracker_overrides.get("match_tentative") is not None:
            tracker.match_iou_thrs.tentative = float(tracker_overrides["match_tentative"])
        if tracker_overrides.get("retain") is not None:
            tracker.num_frames_retain = int(tracker_overrides["retain"])
    if test_scale is not None:
        # The official configs use MultiScaleFlipAug as the second test step.
        config.data.test.pipeline[1].img_scale = tuple(int(v) for v in test_scale)
    config.model.detector.pop("init_cfg", None)
    print(f"building {frontend}: construct model", flush=True)
    model = build_model(config.model)
    # Detector checkpoints were trained outside the ByteTrack wrapper, so they
    # intentionally have no ``detector.`` key prefix.
    print(f"building {frontend}: load checkpoint {checkpoint}", flush=True)
    load_checkpoint(model.detector, str(checkpoint), map_location="cpu")
    model.cfg = config
    print(f"building {frontend}: move to {device}", flush=True)
    model.to(device).eval()
    print(f"building {frontend}: ready", flush=True)
    return model


def inference_frame(model, image_path: Path, frame_id: int):
    from mmcv.parallel import collate, scatter
    from mmdet.datasets.pipelines import Compose

    data = dict(img_info=dict(filename=str(image_path), frame_id=frame_id), img_prefix=None)
    data = Compose(model.cfg.data.test.pipeline)(data)
    data = collate([data], samples_per_gpu=1)
    device = next(model.parameters()).device
    data = scatter(data, [device])[0]
    with torch.inference_mode():
        result, _ = model(return_loss=False, rescale=True, frame_id=frame_id, **data)
    rows = []
    for class_rows in result["track_bboxes"]:
        if len(class_rows):
            rows.extend(np.asarray(class_rows)[:, :5].tolist())
    return rows


def main():
    faulthandler.enable()
    faulthandler.dump_traceback_later(60, repeat=False)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frontend", required=True, choices=("carafe_bytetrack", "autoassign_bytetrack"))
    parser.add_argument("--data", default="data/datafull/test")
    parser.add_argument("--out", required=True)
    parser.add_argument("--scenes", default="", help="Comma-separated scene IDs; empty runs all test scenes")
    parser.add_argument("--max-frames", type=int, default=0, help="Smoke-test limit; zero runs every frame")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    selected = {value.strip() for value in args.scenes.split(",") if value.strip()}
    view1 = Path(args.data) / "1"
    scenes = sorted((p.name[:-2] for p in view1.glob("*-1") if p.is_dir()), key=int)
    if selected:
        scenes = [scene for scene in scenes if scene in selected]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    model = build(args.frontend, args.device)

    for scene in scenes:
        for view in (1, 2):
            frame_dir = Path(args.data) / str(view) / f"{scene}-{view}"
            # MDMT filenames are integer frame numbers without zero padding;
            # lexical sorting would produce 1, 10, 100, ... and invalidate MOT.
            frames = sorted(frame_dir.glob("*.jpg"), key=lambda path: int(path.stem))
            if args.max_frames:
                frames = frames[:args.max_frames]
            model.tracker.reset()
            payload = {}
            for frame_id, image_path in enumerate(frames):
                payload[f"frame={frame_id}"] = inference_frame(model, image_path, frame_id)
                if frame_id % 50 == 0:
                    print(f"{args.frontend} scene={scene} view={view} frame={frame_id}/{len(frames)}", flush=True)
            (out / f"{scene}-{view}.json").write_text(json.dumps(payload), encoding="utf-8")


if __name__ == "__main__":
    main()
