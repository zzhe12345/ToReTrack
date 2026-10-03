"""Train the method's AutoAssign detector with native PyTorch autograd.

This loop uses MMDetection's AutoAssign model/loss and the original image
pipeline. It avoids obsolete OpenMMLab runner/AMP hooks on current PyTorch.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
import random
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "third_party/mia_net_official")]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--initial-weights", default="")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=0, help="Override resize for smoke tests")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        parser.error("epochs and batch size must be positive")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    from uav_tracking.mmcv_ops_compat import install
    install()
    from run_mia_frontend import fixed_mmcv_config_tempdir
    from mmcv import Config
    from mmcv.parallel import collate, scatter
    from mmcv.runner import load_checkpoint
    from mmdet.datasets import build_dataset
    from mmdet.models import build_detector
    with fixed_mmcv_config_tempdir():
        config = Config.fromfile(str(ROOT / "third_party/mia_net_official/configs/mot/bytetrack/bytetrack_autoassign_full_mdmt-private-half.py"))
    detector_config = deepcopy(config.model.detector)
    detector_config.pop("init_cfg", None)
    detector_config.backbone.init_cfg = None
    # Random initialization never triggers an implicit weight download.
    # Optional initial checkpoints must be explicitly supplied by the user.
    model = build_detector(detector_config)
    model.init_weights()
    if args.initial_weights:
        load_checkpoint(model, args.initial_weights, map_location="cpu")
    device = torch.device(args.device)
    model.to(device)
    pipeline = deepcopy(config.train_pipeline)
    if args.image_size:
        next(step for step in pipeline if step.type == "Resize").img_scale = (args.image_size, args.image_size)
    dataset = build_dataset(dict(type="CocoDataset", ann_file=str(args.annotations),
                                 img_prefix="", classes=("pedestrian", "bicycle", "car"),
                                 pipeline=pipeline, filter_empty_gt=True))
    if len(dataset) == 0:
        raise ValueError("No training images with supported labels")
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                                        num_workers=0, collate_fn=lambda batch: collate(batch, samples_per_gpu=args.batch_size))
    optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad],
                                lr=0.002, momentum=0.9, weight_decay=0.0001)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[40, 55], gamma=0.1)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    iteration = 0
    for epoch in range(args.epochs):
        model.train()
        for batch in loader:
            batch = scatter(batch, [device])[0]
            # Match the original linear warm-up; normal learning rates are
            # restored by the epoch scheduler after warm-up completes.
            iteration += 1
            base_lr = 0.002 * (0.1 ** sum(epoch >= boundary for boundary in [40, 55]))
            warmup = min(1.0, 1/5000 + (1-1/5000)*iteration/5000)
            for group in optimizer.param_groups:
                group["lr"] = base_lr * warmup
            optimizer.zero_grad(set_to_none=True)
            losses = model(return_loss=True, **batch)
            loss = sum(value.mean() if isinstance(value, torch.Tensor)
                       else sum(item.mean() for item in value)
                       for name, value in losses.items() if "loss" in name)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite AutoAssign loss at iteration {iteration}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            print(f"AutoAssign epoch={epoch+1}/{args.epochs} batch={iteration} loss={float(loss.detach()):.6f}", flush=True)
        scheduler.step()
        # State-dict keys match the frozen detector loader used at inference.
        torch.save({"state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                    "meta": {"epoch": epoch+1, "CLASSES": dataset.CLASSES},
                    "optimizer": optimizer.state_dict()}, args.out)


if __name__ == "__main__":
    main()
