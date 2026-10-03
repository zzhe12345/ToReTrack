from __future__ import annotations

"""Torchvision/PyTorch adapters for the operators used by this pipeline.

AutoAssign training uses the supported differentiable PyTorch operators.
Unavailable legacy/deformable operators fail explicitly; this module does not
support training every model registered by the upstream OpenMMLab packages.
"""

import sys
import types
import os
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torchvision import ops as tv_ops


class RoIAlign(nn.Module):
    def __init__(self, output_size, spatial_scale=1.0, sampling_ratio=0, pool_mode="avg", aligned=True, **_):
        super().__init__()
        self.output_size, self.spatial_scale = output_size, spatial_scale
        self.sampling_ratio, self.aligned = sampling_ratio, aligned

    def forward(self, features, rois):
        return tv_ops.roi_align(
            features, rois, self.output_size, self.spatial_scale,
            self.sampling_ratio, self.aligned,
        )


class RoIPool(nn.Module):
    def __init__(self, output_size, spatial_scale=1.0, **_):
        super().__init__()
        self.output_size, self.spatial_scale = output_size, spatial_scale

    def forward(self, features, rois):
        return tv_ops.roi_pool(features, rois, self.output_size, self.spatial_scale)


def nms(boxes, scores, iou_threshold=0.5, offset=0, score_threshold=0, max_num=-1, **_):
    if score_threshold > 0:
        valid = scores > score_threshold
        base = torch.nonzero(valid, as_tuple=False).flatten()
        keep_local = tv_ops.nms(boxes[valid], scores[valid], iou_threshold)
        keep = base[keep_local]
    else:
        keep = tv_ops.nms(boxes, scores, iou_threshold)
    if max_num > 0:
        keep = keep[:max_num]
    return torch.cat((boxes[keep], scores[keep, None]), dim=1), keep


def batched_nms(boxes, scores, idxs, nms_cfg, class_agnostic=False):
    threshold = float(nms_cfg.get("iou_threshold", nms_cfg.get("iou_thr", 0.5)))
    keep = tv_ops.nms(boxes, scores, threshold) if class_agnostic else tv_ops.batched_nms(boxes, scores, idxs, threshold)
    max_num = int(nms_cfg.get("max_num", -1))
    if max_num > 0:
        keep = keep[:max_num]
    return torch.cat((boxes[keep], scores[keep, None]), dim=1), keep


def sigmoid_focal_loss(pred, target, gamma=2.0, alpha=0.25, weight=None, reduction="mean"):
    loss = tv_ops.sigmoid_focal_loss(pred, target, alpha=alpha, gamma=gamma, reduction="none")
    if weight is not None:
        loss = loss * weight
    if reduction == "sum":
        return loss.sum()
    if reduction == "mean":
        return loss.mean()
    return loss


class CARAFEPack(nn.Module):
    """Numerically direct, chunked CARAFE implementation for inference."""

    def __init__(self, channels, scale_factor, up_kernel=5, up_group=1,
                 encoder_kernel=3, encoder_dilation=1, compressed_channels=64):
        super().__init__()
        if channels % up_group:
            raise ValueError("channels must be divisible by up_group")
        self.channels, self.scale_factor = channels, scale_factor
        self.up_kernel, self.up_group = up_kernel, up_group
        self.channel_compressor = nn.Conv2d(channels, compressed_channels, 1)
        self.content_encoder = nn.Conv2d(
            compressed_channels, up_kernel * up_kernel * up_group * scale_factor * scale_factor,
            encoder_kernel, padding=(encoder_kernel - 1) * encoder_dilation // 2,
            dilation=encoder_dilation,
        )

    def init_weights(self):
        return None

    def kernel_normalizer(self, mask):
        mask = F.pixel_shuffle(mask, self.scale_factor)
        n, channels, height, width = mask.shape
        mask = mask.view(n, channels // (self.up_kernel ** 2), self.up_kernel ** 2, height, width)
        return F.softmax(mask, dim=2).view(n, channels, height, width).contiguous()

    def feature_reassemble(self, x, mask):
        n, channels, height, width = x.shape
        scale, kernel, groups = self.scale_factor, self.up_kernel, self.up_group
        out_h, out_w = height * scale, width * scale
        patches = F.unfold(x, kernel_size=kernel, padding=kernel // 2)
        patches = patches.view(n, groups, channels // groups, kernel * kernel, height * width)
        mask = mask.view(n, groups, kernel * kernel, out_h * out_w)
        y = torch.arange(out_h, device=x.device).div(scale, rounding_mode="floor")
        x_index = torch.arange(out_w, device=x.device).div(scale, rounding_mode="floor")
        source = (y[:, None] * width + x_index[None, :]).reshape(-1)
        output = x.new_empty((n, channels, out_h * out_w))
        chunk = 4096
        for start in range(0, out_h * out_w, chunk):
            stop = min(start + chunk, out_h * out_w)
            selected = patches.index_select(-1, source[start:stop])
            values = (selected * mask[..., start:stop].unsqueeze(2)).sum(dim=3)
            output[..., start:stop] = values.reshape(n, channels, stop - start)
        return output.view(n, channels, out_h, out_w)

    def forward(self, x):
        mask = self.kernel_normalizer(self.content_encoder(self.channel_compressor(x)))
        return self.feature_reassemble(x, mask)


class _UnavailableOp(nn.Module):
    def __init__(self, *_, **__):
        super().__init__()

    def forward(self, *_, **__):
        raise RuntimeError("This MMCV operator is unavailable in the inference compatibility layer")


def _unavailable(*_, **__):
    raise RuntimeError("This MMCV operator is unavailable in the inference compatibility layer")


def install() -> None:
    if getattr(sys.modules.get("mmcv.ops"), "_uav_compat", False):
        return
    # MMCV imports YAPF, whose grammar loader writes a cache under the Windows
    # LocalAppData known folder.  That location is read-only in the workspace
    # sandbox and Python's NamedTemporaryFile retries there for minutes.  Route
    # only YAPF/platformdirs cache requests to a deterministic writable folder
    # before importing MMCV; this does not alter model or inference numerics.
    import platformdirs

    cache_root = Path(os.environ.get("MIA_YAPF_CACHE", "tmp/mia_yapf_cache")).resolve()
    cache_root.mkdir(parents=True, exist_ok=True)

    def workspace_user_cache_dir(appname=None, appauthor=None, version=None, *_, **__):
        path = cache_root
        if appname:
            path /= str(appname)
        if version:
            path /= str(version)
        path.mkdir(parents=True, exist_ok=True)
        return str(path)

    platformdirs.user_cache_dir = workspace_user_cache_dir
    import mmcv.cnn

    module = types.ModuleType("mmcv.ops")
    module.__path__ = []
    module._uav_compat = True
    concrete = {
        "RoIAlign": RoIAlign, "RoIPool": RoIPool, "nms": nms,
        "batched_nms": batched_nms, "sigmoid_focal_loss": sigmoid_focal_loss,
        "CARAFEPack": CARAFEPack,
    }
    for name, value in concrete.items():
        setattr(module, name, value)
    for name in (
        "CornerPool", "DeformConv2d", "MaskedConv2d", "ModulatedDeformConv2d",
        "DeformRoIPool", "DeformRoIPoolPack", "ModulatedDeformRoIPoolPack",
    ):
        setattr(module, name, _UnavailableOp)
    for name in ("deform_conv2d", "point_sample", "rel_roi_point_to_rel_img_point", "nms_match", "get_onnxruntime_op_path"):
        setattr(module, name, _unavailable)
    carafe_module = types.ModuleType("mmcv.ops.carafe")
    carafe_module.CARAFEPack = CARAFEPack
    carafe_module.CARAFE = carafe_module.CARAFENaive = _UnavailableOp
    carafe_module.carafe = carafe_module.carafe_naive = _unavailable
    roi_align_module = types.ModuleType("mmcv.ops.roi_align")
    roi_align_module.roi_align = lambda features, rois, output_size, spatial_scale=1.0, sampling_ratio=0, pool_mode="avg", aligned=True: tv_ops.roi_align(
        features, rois, output_size, spatial_scale, sampling_ratio, aligned
    )
    nms_module = types.ModuleType("mmcv.ops.nms")
    nms_module.nms, nms_module.batched_nms = nms, batched_nms
    merge_module = types.ModuleType("mmcv.ops.merge_cells")
    merge_module.GlobalPoolingCell = merge_module.SumCell = merge_module.ConcatCell = _UnavailableOp
    deform_module = types.ModuleType("mmcv.ops.modulated_deform_conv")
    deform_module.ModulatedDeformConv2d = _UnavailableOp
    msda_module = types.ModuleType("mmcv.ops.multi_scale_deform_attn")
    msda_module.MultiScaleDeformableAttention = _UnavailableOp
    sys.modules["mmcv.ops"] = module
    sys.modules["mmcv.ops.carafe"] = carafe_module
    sys.modules["mmcv.ops.roi_align"] = roi_align_module
    sys.modules["mmcv.ops.nms"] = nms_module
    sys.modules["mmcv.ops.merge_cells"] = merge_module
    sys.modules["mmcv.ops.modulated_deform_conv"] = deform_module
    sys.modules["mmcv.ops.multi_scale_deform_attn"] = msda_module
    mmcv.ops = module

    registry = mmcv.cnn.UPSAMPLE_LAYERS
    if registry.get("carafe") is None:
        registry.register_module(name="carafe", module=CARAFEPack)
