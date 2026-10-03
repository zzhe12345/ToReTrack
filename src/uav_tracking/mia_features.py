"""Frozen detector FPN descriptors and deterministic geometry projection."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.ops import roi_align


def prepare(model, pipeline, image_path: Path):
    from mmcv.parallel import collate, scatter
    data = dict(img_info=dict(filename=str(image_path), frame_id=0), img_prefix=None)
    data = pipeline(data)
    data = scatter(collate([data], samples_per_gpu=1), [next(model.parameters()).device])[0]
    image = data["img"][0] if isinstance(data["img"], (list, tuple)) else data["img"]
    metas = data["img_metas"][0] if isinstance(data["img_metas"], (list, tuple)) else data["img_metas"]
    while isinstance(metas, (list, tuple)):
        metas = metas[0]
    return image, metas


@torch.inference_mode()
def frame_features(model, pipeline, image_path: Path, boxes: list[list[float]],
                   pooling: str = "per_level_roi_align_1x1_then_equal_fpn_mean") -> torch.Tensor:
    image, meta = prepare(model, pipeline, image_path)
    pyramid = model.detector.extract_feat(image)
    # FPN tensors are defined over the padded detector input.  Using img_shape
    # here shifts the sampling grid whenever Pad(size_divisor=32) adds pixels.
    # pad_shape gives the exact strides of the five AutoAssign FPN maps.
    padded_h, padded_w = meta["pad_shape"][:2]
    scale = np.asarray(meta.get("scale_factor", [1, 1, 1, 1]), dtype=np.float32)
    scaled = np.asarray(boxes, dtype=np.float32) * scale[None]
    scaled[:, [0, 2]] = np.clip(scaled[:, [0, 2]], 0, padded_w - 1)
    scaled[:, [1, 3]] = np.clip(scaled[:, [1, 3]], 0, padded_h - 1)
    scaled[:, 2] = np.maximum(scaled[:, 2], scaled[:, 0] + 1.0)
    scaled[:, 3] = np.maximum(scaled[:, 3], scaled[:, 1] + 1.0)
    batch_indices = np.zeros((len(scaled), 1), dtype=np.float32)
    rois = torch.as_tensor(np.concatenate([batch_indices, scaled], axis=1), device=image.device)
    if pooling == "per_level_roi_align_1x1_then_equal_fpn_mean":
        levels = []
        for feature in pyramid:
            _, _, height, width = feature.shape
            # The detector input is padded to an exact multiple of every FPN
            # stride, so width / padded_w is the precise spatial scale.
            pooled = roi_align(
                feature, rois, output_size=(1, 1), spatial_scale=width / padded_w,
                sampling_ratio=2, aligned=True,
            ).flatten(1)
            levels.append(pooled)
        return F.normalize(torch.stack(levels).mean(0), dim=1).cpu()
    if pooling != "assigned_fpn_roi_align_3x3_meanmax":
        raise ValueError(f"unknown frozen-FPN pooling protocol: {pooling}")

    # Standard FPN scale assignment avoids averaging semantically mismatched
    # P3--P7 regions.  A 3x3 mean/max descriptor retains both regional context
    # and the strongest local activation while keeping the backbone frozen.
    widths = torch.as_tensor(scaled[:, 2] - scaled[:, 0], device=image.device)
    heights = torch.as_tensor(scaled[:, 3] - scaled[:, 1], device=image.device)
    canonical = torch.floor(4.0 + torch.log2(torch.sqrt(widths * heights) / 224.0 + 1e-6))
    level_indices = canonical.clamp(3, 3 + len(pyramid) - 1).long() - 3
    channels = int(pyramid[0].shape[1])
    descriptor = pyramid[0].new_zeros((len(rois), channels * 2))
    for level, feature in enumerate(pyramid):
        selected = torch.nonzero(level_indices == level, as_tuple=False).flatten()
        if not len(selected):
            continue
        _, _, _height, width = feature.shape
        pooled = roi_align(
            feature, rois[selected], output_size=(3, 3), spatial_scale=width / padded_w,
            sampling_ratio=2, aligned=True,
        )
        mean = pooled.mean(dim=(-2, -1))
        maximum = pooled.amax(dim=(-2, -1))
        descriptor[selected] = torch.cat([mean, maximum], dim=1)
    return F.normalize(descriptor, dim=1).cpu()
