# YOLOv5 common modules
# Copyright (c) Alibaba, Inc. and its affiliates.

import math
from copy import copy
from pathlib import Path
import numpy as np
import pandas as pd
import requests
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from PIL import Image
from torch.cuda import amp
# from dcn_v2 import DCN as _dcn

from utils.datasets import letterbox
from utils.general import non_max_suppression, make_divisible, scale_coords, increment_path, xyxy2xywh, save_one_box, box_iou
from utils.plots import colors, plot_one_box
from utils.torch_utils import time_synchronized


def autopad(k, p=None):  # kernel, padding
    # Pad to 'same'
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


# def DWConv(c1, c2, k=1, s=1, act=True):
#     # Depthwise convolution
#     return Conv(c1, c2, k, s, g=math.gcd(c1, c2), act=act)


class Conv(nn.Module):
    # Standard convolution
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True, d=1):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), dilation=d, groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x, act=True):
        return self.act(self.conv(x)) if act else self.conv(x)


class DWConv(Conv):
    # Depthwise convolution
    def __init__(self, c1, c2, k=1, s=1, act=True):  # ch_in, ch_out, kernel, stride
        super(DWConv, self).__init__(c1, c2, k, s, g=math.gcd(c1, c2), act=act)


class DCN(Conv):
    # Deformable convolution
    def __init__(self, c1, c2, k=1, s=1, act=True):  # ch_in, ch_out, kernel, stride
        super(DCN, self).__init__(c1, c2, k, s, act=act)
        self.conv = _dcn(c1, c2, k, s, autopad(k, None), dilation=1)


class ResBottleneck(nn.Module):
    # Bottleneck block in ResNet models
    def __init__(self, c1, c2, s=1, p=None, g=1, act=True, d=1):  # ch_in, ch_out, kernel, stride, padding, groups
        super(ResBottleneck, self).__init__()
        e = 4
        c = c2 // e
        self.conv1 = nn.Conv2d(c1, c, 1, 1, autopad(1, p), dilation=d, groups=g, bias=False)
        self.bn1 = nn.BatchNorm2d(c)
        self.conv2 = nn.Conv2d(c, c, 3, s, autopad(3, p), dilation=d, groups=g, bias=False)
        self.bn2 = nn.BatchNorm2d(c)
        self.conv3 = nn.Conv2d(c, c2, 1, 1, autopad(1, p), dilation=d, groups=g, bias=False)
        self.bn3 = nn.BatchNorm2d(c2)
        if s != 1 or c1 != c2:
            self.downsample = nn.Sequential(
                nn.Conv2d(c1, c2, 1, s, bias=False),
                nn.BatchNorm2d(c2),
            )
        else:
            self.downsample = None
        self.act = nn.ReLU(inplace=True) if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.act(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.act(out)

        return out


class ResBlockLayer(nn.Module):
    # ResBlockLayer with n ResBottleNeck blocks
    def __init__(self, c1, c2, s=1, n=1, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super(ResBlockLayer, self).__init__()
        self.m = nn.Sequential(
            ResBottleneck(c1, c2, s, g=g),
            *[ResBottleneck(c2, c2, g=g) for _ in range(n-1)]
        )

    def forward(self, x):
        return self.m(x)


class DepthwiseSeparableConvModule(nn.Module):
    def __init__(self, c1, c2, k, s=1, p=None, d=1):
        super(DepthwiseSeparableConvModule, self).__init__()
        # depthwise convolution
        self.depthwise_conv = Conv(c1, c1, k, s, p, g=c1, d=d)
        self.pointwise_conv = Conv(c1, c2, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)
        return x


class RTMDetCSPNeXtBlock(nn.Module):
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 5), e=0.5):
        super(RTMDetCSPNeXtBlock, self).__init__()
        c = int(c2 * e)
        self.conv1 = Conv(c1, c, k[0], 1, g=g)
        self.conv2 = DepthwiseSeparableConvModule(c, c2, k[1], 1)
        self.add_identity = \
            shortcut and c1 == c2

    def forward(self, x):
        out = self.conv2(self.conv1(x))
        return (out + x) if self.add_identity else out


class RTMDetChannelAttention(nn.Module):
    def __init__(self, channels: int):
        super(RTMDetChannelAttention, self).__init__()
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Conv2d(channels, channels, 1, 1, 0, bias=True)
        if torch.__version__ < "1.7.0":
            self.act = nn.Hardsigmoid()
        else:
            self.act = nn.Hardsigmoid(inplace=True)

    def forward(self, x):
        with torch.cuda.amp.autocast(enabled=False):
            out = self.global_avgpool(x)
        return x * self.act(self.fc(out))


class RTMDetCSPLayer(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=True, ca=True, e=0.5):
        super(RTMDetCSPLayer, self).__init__()
        c = int(c2 * e)
        self.ca = ca
        self.main_conv = Conv(c1, c, 1, 1)
        self.short_conv = Conv(c1, c, 1, 1)
        self.final_conv = Conv(2*c, c2, 1, 1)

        self.blocks = nn.Sequential(*[
            RTMDetCSPNeXtBlock(c, c, shortcut, e=1.0) for _ in range(n)
        ])
        if ca:
            self.attention = RTMDetChannelAttention(2 * c)

    def forward(self, x):
        x_short = self.short_conv(x)

        x_main = self.main_conv(x)
        x_main = self.blocks(x_main)

        x_final = torch.cat((x_main, x_short), dim=1)

        if self.ca:
            x_final = self.attention(x_final)
        return self.final_conv(x_final)


class TransformerLayer(nn.Module):
    # Transformer layer https://arxiv.org/abs/2010.11929 (LayerNorm layers removed for better performance)
    def __init__(self, c, num_heads, norm=False):
        super().__init__()
        self.q = nn.Linear(c, c, bias=False)
        self.k = nn.Linear(c, c, bias=False)
        self.v = nn.Linear(c, c, bias=False)
        self.ma = nn.MultiheadAttention(embed_dim=c, num_heads=num_heads)
        self.fc1 = nn.Linear(c, c, bias=False)
        self.fc2 = nn.Linear(c, c, bias=False)
        if norm:
            self.norm1 = nn.LayerNorm(c)
            self.norm2 = nn.LayerNorm(c)
        else:
            self.norm1 = None
            self.norm2 = None

    def forward(self, x, pos_embed=None):
        if pos_embed is not None:
            q, k, v = self.q(x), self.k(x + pos_embed), self.v(x + pos_embed)
        else:
            q, k, v = self.q(x), self.k(x), self.v(x)
        x = self.ma(q, k, v)[0] + x
        if self.norm1 is not None:
            x = self.norm1(x)
        x = self.fc2(self.fc1(x)) + x
        if self.norm2 is not None:
            x = self.norm2(x)
        return x


class TransformerBlock(nn.Module):
    # Vision Transformer https://arxiv.org/abs/2010.11929
    def __init__(self, c1, c2, num_heads, num_layers):
        super().__init__()
        self.conv = None
        if c1 != c2:
            self.conv = Conv(c1, c2)
        self.linear = nn.Linear(c2, c2)  # learnable position embedding
        self.tr = nn.Sequential(*[TransformerLayer(c2, num_heads) for _ in range(num_layers)])
        self.c2 = c2

    def forward(self, x):
        if self.conv is not None:
            x = self.conv(x)
        b, _, w, h = x.shape
        p = x.flatten(2)         # [b, c, w*h]
        p = p.unsqueeze(0)       # [1, b, c, w*h]
        p = p.transpose(0, 3)    # [w*h, b, c, 1]
        p = p.squeeze(3)         # [w*h, b, c]
        e = self.linear(p)       # [w*h, b, c]
        x = p + e                # [w*h, b, c]

        x = self.tr(x)           # [w*h, b, c]
        x = x.unsqueeze(3)       # [w*h, b, c, 1]
        x = x.transpose(0, 3)    # [1, b, c, w*h]
        x = x.reshape(b, self.c2, w, h)
        return x


class MaskedTransformerBlock(nn.Module):
    # Vision Transformer https://arxiv.org/abs/2010.11929
    def __init__(self, c1, c2, num_heads, num_layers):
        super().__init__()
        self.conv = None
        if c1 != c2:
            self.conv = Conv(c1, c2)
        self.e = None  # fixed position embedding
        self.tr = nn.Sequential(*[TransformerLayer(c2, num_heads) for _ in range(num_layers)])
        self.c2 = c2

    def forward(self, x, mask):
        if self.conv is not None:
            x = self.conv(x)
        b, c, h, w = x.shape
        assert b == 1, 'multi-batch is not supported'
        if self.e is None or self.e.shape[1] != h * w:
            self.e = self.build_2d_sincos_position_embedding(w, h, c, dtype=x.dtype, device=x.device)
        
        mw = mask.shape[-1]
        if mw != w:
            s = int(mw // w)
            mask = F.avg_pool2d(mask.float(), s, s, 0) > 0.5
        
        p = x.flatten(2)         # [b, c, w*h]
        p = p.transpose(1, 2)    # [b, w*h, c]
        p0 = p.detach()          # [b, w*h, c]
        z = p + self.e           # [b, w*h, c]

        m = mask.flatten(1)      # [b, w*h]
        # max_len = x.sum(1).max()
        z = z[m][:, None, :]     # [n, b, c]

        z = self.tr(z)           # [n, b, c]
        z = z.transpose(0, 1)    # [b, n, c]
        p0[m] = z.reshape(-1, c) # [b, w*h, c]
        x = x.transpose(1, 2)    # [b, c, w*h]
        x = x.reshape(b, self.c2, h, w)
        return x

    @staticmethod
    def build_2d_sincos_position_embedding(w, h, embed_dim=256, temperature=10000., dtype=torch.float32, device='cuda'):
        grid_w = torch.arange(int(w), dtype=torch.float32)
        grid_h = torch.arange(int(h), dtype=torch.float32)
        grid_w, grid_h = torch.meshgrid(grid_w, grid_h)  # meshgrid() got an unexpected keyword argument 'indexing'
        assert embed_dim % 4 == 0, \
            'Embed dimension must be divisible by 4 for 2D sin-cos position embedding'
        pos_dim = embed_dim // 4
        omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim
        omega = 1. / (temperature ** omega)

        out_w = grid_w.flatten()[..., None] @ omega[None]
        out_h = grid_h.flatten()[..., None] @ omega[None]

        # shape(1,w*h,c)
        return torch.cat([torch.sin(out_w), torch.cos(out_w),
                          torch.sin(out_h), torch.cos(out_h)], axis=1)[None, :, :].to(dtype).to(device)


class HeatMapParser(nn.Module):
    def __init__(self, c, ratio=8, threshold=0.5, mask_only=False, cluster_only=False):
        super().__init__()
        self.c = c
        self.ratio = ratio
        self.threshold = threshold
        self.mask_only = mask_only
        self.cluster_only = cluster_only

        self.grid = None
    
    def forward(self, x):
        x, heatmaps = x
        bs, c, ny, nx = x.shape
        device, dtype = x.device, x.dtype
        assert c == self.c, f'{c} - {self.c}'
        assert len(heatmaps) <= 3
        mask_pred = heatmaps[0].detach()
        if torch.max(mask_pred) > 1. or torch.min(mask_pred) < 0.:
            mask_pred = mask_pred.sigmoid()

        mask_pred = mask_pred[:, 0, :, :].detach() 

        if getattr(self, 'mask_only', False):
            return x, self.threshold
        
        # t0 = time_synchronized()
        if self.training:
            # if getattr(self, 'cluster_only', False):
            #     total_clusters = self.ada_slicer(mask_pred, self.ratio, self.threshold * 1. + 0.)
            #     return self.get_offsets_by_clusters(total_clusters).to(device)
            return self.uni_slicer(x, mask_pred, self.ratio, self.threshold * 1. + 0., device=device)
        else:
            # total_clusters = self.ada_slicer(mask_pred, self.ratio, self.threshold * 1.0 + 0.)
            total_clusters = self.ada_slicer_fast(mask_pred, self.ratio, self.threshold * 1.0 + 0.)
            if getattr(self, 'cluster_only', False):
                return self.get_offsets_by_clusters(total_clusters).to(device)
            # t1 = time_synchronized()
            patches, offsets = [], []
            for bi, clusters in enumerate(total_clusters):
                for x1, y1, x2, y2 in clusters:
                    patches.append(x[bi, :, y1:y2, x1:x2])
                    offsets.append(torch.tensor([bi, x1, y1, x2, y2]))
                    assert patches[-1].shape[-2] == y2 - y1 and patches[-1].shape[-1] == x2 - x1, '%f %f %f %f' % (x1, y1, x2, y2)
            # t2 = time_synchronized()
            # print(f'Patchify: {t1 - t0:.3f}s. Slice:{t2 - t1:.3f}s')

            if len(patches):
                return torch.stack(patches), torch.stack(offsets).to(device)
            else:
                return torch.zeros((0, c, ny, nx), device=device), torch.zeros((0, 5), device=device)
    
    @staticmethod
    def get_offsets_by_clusters(total_clusters):
        offsets = []
        for bi, clusters in enumerate(total_clusters):
            b = torch.full_like(clusters[:, :1], bi)
            offsets.append(torch.cat((b, clusters), dim=1))
        return torch.cat(offsets)

    @torch.no_grad()
    def ada_slicer(self, mask_pred: torch.Tensor, ratio=8, threshold=0.3):   # better     
        # t0 = time_synchronized()
        bs, height, width = mask_pred.shape
        device, dtype = mask_pred.device, mask_pred.dtype
        cluster_wh = max(make_divisible(width / ratio, 4), make_divisible(height / ratio, 4))  # 保证正方形
        cluster_w, cluster_h = cluster_wh, cluster_wh
        # cluster_w, cluster_h = make_divisible(width / ratio, 4), make_divisible(height / ratio, 4)
        half_clus_w,  half_clus_h = cluster_w // 2, cluster_h // 2
        outs = []

        # t1 = time_synchronized()
        activated = mask_pred >= threshold
        maxima = F.max_pool2d(mask_pred, 3, stride=1, padding=1) == mask_pred
        obj_centers = activated & maxima
        padding = half_clus_w // 2
        obj_sizes = F.avg_pool2d(mask_pred, padding * 2 + 1, stride=1, padding=padding)
        
        # bi, yi, xi
        # t2 = time_synchronized()
        cb, cy, cx = obj_centers.nonzero(as_tuple=True)
        obj_sizes = obj_sizes[cb, cy, cx]

        # t3 = time_synchronized()
        for bi in range(bs):
            ci = cb == bi
            cn = ci.sum().item()
            if cn == 0:
                outs.append(torch.zeros((0, 4), device=device))
                continue

            if bs == 1:
                sizes = obj_sizes
                cy_bi, cx_bi = cy, cx
            else:
                sizes = obj_sizes[ci]
                cy_bi, cx_bi = cy[ci], cx[ci]
                
            # shape(n,1)
            init_x1 = cx_bi.clamp(half_clus_w, width - half_clus_w) - half_clus_w
            init_y1 = cy_bi.clamp(half_clus_h, height - half_clus_h) - half_clus_h

            # shape(1,m)
            if not hasattr(self, 'grid') or self.grid is None or self.grid[0].shape[-1] != cluster_h*cluster_w:
                gy, gx = torch.meshgrid(torch.arange(cluster_h), torch.arange(cluster_w))
                self.grid = (gy.reshape(1, -1).to(device), gx.reshape(1, -1).to(device))
            gy, gx = self.grid

            # shape(n,m)
            act_x, act_y = (init_x1.view(-1, 1) + gx).view(-1), (init_y1.view(-1, 1) + gy).view(-1)
            act = activated[bi, act_y, act_x].view(cn, cluster_h, cluster_w)
            
            # t4 = time_synchronized()
            act_x, act_y = act.any(dim=1).long(), act.any(dim=2).long()
            dx1, dx2 = (1 - act_x).argmin(dim=1), -(1 - act_x.flip((1,))).argmin(dim=1)
            dy1, dy2 = (1 - act_y).argmin(dim=1), -(1 - act_y.flip((1,))).argmin(dim=1)
            dx = torch.where(dx1.abs() > dx2.abs(), dx1, dx2)
            dy = torch.where(dy1.abs() > dy2.abs(), dy1, dy2)
            
            # t5 = time_synchronized()
            refine_x1, refine_y1 = (init_x1 + dx).clamp(0, width - cluster_w).to(dtype), \
                                    (init_y1 + dy).clamp(0, height - cluster_h).to(dtype)
            refine_x2, refine_y2 = refine_x1 + cluster_w, refine_y1 + cluster_h
            total_clusters = torch.stack((refine_x1, refine_y1, refine_x2, refine_y2), dim=1).long()
            
            # i = torchvision.ops.nms(total_clusters, sizes, 0.8)  # NMS
            # clusters = total_clusters[i].long()

            # t6 = time_synchronized()
            overlap = (refine_x1[:, None] <= cx_bi[None, :]) & (cx_bi[None, :] < refine_x2[:, None]) & \
                      (refine_y1[:, None] <= cy_bi[None, :]) & (cy_bi[None, :] < refine_y2[:, None])
            clusters = []
            contained = torch.full_like(overlap[0], False)
            for max_i in torch.argsort(sizes, descending=True):
                if contained[max_i]:
                    continue
                clusters.append(total_clusters[max_i])
                contained |= overlap[max_i]

            # t7 = time_synchronized()
            outs.append(torch.stack(clusters) if len(clusters) else torch.zeros_like(total_clusters[:0, :]))
    
            # print(f't1: {(t1-t0)*1000:.3f}, t2: {(t2-t1)*1000:.3f}, t3: {(t3-t2)*1000:.3f}, t4: {(t4-t3)*1000:.3f}, t5: {(t5-t4)*1000:.3f}, t6: {(t6-t5)*1000:.3f}, t7: {(t7-t6)*1000:.3f}')
        return outs
    
    @torch.no_grad()
    def ada_slicer_fast(self, mask_pred: torch.Tensor, ratio=8, threshold=0.3):   # faster
        # t0 = time_synchronized()
        bs, height, width = mask_pred.shape
        # assert width % ratio == 0 and height % ratio == 0, f'{width} // {height}'
        device, dtype = mask_pred.device, mask_pred.dtype
        # cluster_wh = max(make_divisible(width / ratio, 4), make_divisible(height / ratio, 4))  # 保证正方形
        # cluster_w, cluster_h = cluster_wh, cluster_wh
        cluster_w, cluster_h = make_divisible(width / ratio, 4), make_divisible(height / ratio, 4)
        # cluster_w, cluster_h = width // ratio, height // ratio
        # assert cluster_w % 4 == 0 and cluster_h % 4 == 0, f'{width} -> {cluster_w} // {height} -> {cluster_h}'
        ratio_x, ratio_y = int(math.ceil(width / cluster_w)), int(math.ceil(height / cluster_h))
        half_clus_w,  half_clus_h = cluster_w // 2, cluster_h // 2
        outs = []

        if getattr(self, 'grid_vtx', None) is None or self.grid_vtx.size(0) != ratio_x*ratio_y*bs:
            gy, gx = torch.meshgrid(torch.arange(ratio_y), torch.arange(ratio_x))
            gxy = torch.stack((gy.reshape(-1), gx.reshape(-1)), dim=1).unsqueeze(0).repeat(bs, 1, 1).view(-1, 2)  # shape(bs*8*8,2)
            gb = torch.arange(bs).view(-1, 1).repeat(1, ratio_x * ratio_y).view(-1, 1)  # shape(bs*8*8, 1)
            self.grid_vtx = torch.cat((gb, gxy), dim=1).to(device)  # shape(bs*8*8, 3)
        rb, ry, rx = self.grid_vtx.T

        if getattr(self, 'grid', None) is None or self.grid[0].shape[-1] != cluster_h*cluster_w:
            gy, gx = torch.meshgrid(torch.arange(cluster_h), torch.arange(cluster_w))
            self.grid = (gy.reshape(1, -1).to(device), gx.reshape(1, -1).to(device))
        gy, gx = self.grid

        # t1 = time_synchronized()
        activated = mask_pred >= threshold
        maxima: torch.Tensor = F.max_pool2d(mask_pred, 3, stride=1, padding=1) == mask_pred
        obj_centers = activated & maxima
        if (~obj_centers).all():
            return [torch.zeros((0, 4), device=device) for _ in range(bs)]
        padding = max(half_clus_w, half_clus_h) // 2
        obj_sizes = F.avg_pool2d(mask_pred, padding * 2 + 1, stride=1, padding=padding)
        
        valid_regions = F.pad(obj_centers, (0, ratio_x*cluster_w-width, 0, ratio_y*cluster_h-height))
        valid_regions = F.max_pool2d(valid_regions.float(), (cluster_h, cluster_w), stride=(cluster_h, cluster_w), padding=0)
        valid_regions = valid_regions.view(-1) > 0
        cb, x1, y1 = rb[valid_regions], rx[valid_regions] * cluster_w, ry[valid_regions] * cluster_h

        act_x, act_y = (x1.view(-1, 1) + gx).view(-1), (y1.view(-1, 1) + gy).view(-1)
        act_b = cb.view(-1, 1).repeat((1, gy.size(1))).view(-1)
        activated = F.pad(activated, (0, ratio_x*cluster_w-width, 0, ratio_y*cluster_h-height))
        act = activated[act_b, act_y, act_x].view(cb.shape[0], cluster_h, cluster_w)
                
        act_x, act_y = act.any(dim=1).long(), act.any(dim=2).long()  # shape(nc, cw), shape(nc, ch)
        dx1, dx2 = (1 - act_x).argmin(dim=1), -(1 - act_x.flip((1,))).argmin(dim=1)
        dy1, dy2 = (1 - act_y).argmin(dim=1), -(1 - act_y.flip((1,))).argmin(dim=1)
        dx = torch.where(dx1.abs() > dx2.abs(), dx1, dx2)
        dy = torch.where(dy1.abs() > dy2.abs(), dy1, dy2)
        
        # t5 = time_synchronized()
        x1, y1 = (x1 + dx).clamp(0, width - cluster_w), \
                 (y1 + dy).clamp(0, height - cluster_h)
        x2, y2 = x1 + cluster_w, y1 + cluster_h
        bboxes = torch.stack((x1, y1, x2, y2), dim=1).long()

        # offsets = (cb * max(width, height)).unsqueeze(1)
        # scores = obj_sizes[cb, y1 + half_clus_h, x1 + half_clus_w]
        # indices = torchvision.ops.nms((bboxes + offsets).float(), scores, iou_threshold=0.9)  # 0.65
        # cb, bboxes = cb[indices], bboxes[indices]

        for bi in range(bs):
            outs.append(bboxes[cb == bi])

        return outs

    def uni_slicer(self, feat, mask_pred, ratio=8, threshold=0.3, device='cuda'):
        def _slice(x: torch.Tensor):
            # if len(x.shape) == 4:
            #     b, c, h, w = x.shape
            #     return x.view(b, c, ratio, h//ratio, ratio, w//ratio).permute(0,2,4,1,3,5).contiguous().view(b*ratio*ratio, c, h//ratio, w//ratio)
            # else:
            #     b, h, w = x.shape
            #     return x.view(b, ratio, h//ratio, ratio, w//ratio).transpose(2,3).contiguous().view(b*ratio*ratio, h//ratio, w//ratio)
        
            x_list = torch.chunk(x, ratio, dim=-2)  # [shape(bs,c,h//8,w)] * 8
            y = []
            for x in x_list:
                y.extend(torch.chunk(x, ratio, dim=-1))  # [shape(bs,c,h//8,w//8)] * 8
            return torch.cat(y, dim=0)  # shape(8*8*bs,c,h//8,w//8)

        bs, height, width = mask_pred.shape
        assert height == width
        assert width % (ratio * 4) == 0 and height % (ratio * 4) == 0, f'{width}, {height}'
        cluster_wh = max(make_divisible(width / ratio, 4), make_divisible(height / ratio, 4))  # 保证正方形

        if not hasattr(self, 'grid_off') or len(self.grid_off) != bs * ratio * ratio or self.grid_off.device != device:
            xrange = torch.arange(ratio)
            gy, gx = torch.meshgrid(xrange, xrange)
            gxy = torch.stack((gy.reshape(-1), gx.reshape(-1)), dim=1).unsqueeze(1).repeat(1, bs, 1).view(-1, 2)  # shape(8*8*bs,2)
            gb = torch.arange(bs).view(1, -1).repeat(ratio ** 2, 1).view(-1, 1)  # shape(8*8*bs)
            gy, gx = gxy.T
            grid = torch.stack((gx, gy, gx + 1, gy + 1), dim=-1) * cluster_wh
            self.grid_off = torch.cat((gb, grid), dim=1).to(device)
        
        if getattr(self, 'cluster_only', False):
            return self.grid_off
        
        patches = _slice(feat)  # shape(8*8*bs,c,h//8,w//8)

        return patches, self.grid_off
    
        # activated = mask_pred >= threshold
        # maxima = F.max_pool2d(mask_pred, 3, stride=1, padding=1) == mask_pred
        # obj_centers = activated & maxima

        # mask = _slice(obj_centers)  # shape(8*8*bs,h//8,w//8)
        # indices = mask.view(len(patches), -1).any(dim=1)
        # return patches[indices], self.grid_off[indices]

    
class Bottleneck(nn.Module):
    # Standard bottleneck
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, shortcut, groups, expansion
        super(Bottleneck, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_, c2, 3, 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class BottleneckCSP(nn.Module):
    # CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super(BottleneckCSP, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = nn.Conv2d(c1, c_, 1, 1, bias=False)
        self.cv3 = nn.Conv2d(c_, c_, 1, 1, bias=False)
        self.cv4 = Conv(2 * c_, c2, 1, 1)
        self.bn = nn.BatchNorm2d(2 * c_)  # applied to cat(cv2, cv3)
        self.act = nn.LeakyReLU(0.1, inplace=True)
        self.m = nn.Sequential(*[Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)])

    def forward(self, x):
        y1 = self.cv3(self.m(self.cv1(x)))
        y2 = self.cv2(x)
        return self.cv4(self.act(self.bn(torch.cat((y1, y2), dim=1))))


class C3(nn.Module):
    # CSP Bottleneck with 3 convolutions
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super(C3, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # act=FReLU(c2)
        self.m = nn.Sequential(*[Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)])
        # self.m = nn.Sequential(*[CrossConv(c_, c_, 3, 1, g, 1.0, shortcut) for _ in range(n)])

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1))


class C3TR(C3):
    # C3 module with TransformerBlock()
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = TransformerBlock(c_, c_, 4, n)


class MaskedC3TR(nn.Module):
    # C3 module with TransformerBlock()
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super(MaskedC3TR, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_)
        self.cv2 = Conv(c1, c_)
        self.cv3 = Conv(2 * c_, c2, 1)  # act=FReLU(c2)
        self.m = MaskedTransformerBlock(c_, c_, 4, n)

    def forward(self, x):
        x, masks = x
        mask = masks[0]
        return self.cv3(torch.cat((self.m(self.cv1(x), mask), self.cv2(x)), dim=1))


class C2f(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        """Initialize CSP bottleneck layer with two convolutions with arguments ch_in, ch_out, number, shortcut, groups,
        expansion.
        """
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class DFL(nn.Module):
    """
    Integral module of Distribution Focal Loss (DFL).

    Proposed in Generalized Focal Loss https://ieeexplore.ieee.org/document/9792391
    """

    def __init__(self, c1=16):
        """Initialize a convolutional layer with a given number of input channels."""
        super().__init__()
        self.conv = nn.Conv2d(c1, 1, 1, bias=False).requires_grad_(False)
        x = torch.arange(c1, dtype=torch.float)
        self.conv.weight.data[:] = nn.Parameter(x.view(1, c1, 1, 1))
        self.c1 = c1

    def forward(self, x):
        """Applies a transformer layer on input tensor 'x' and returns a tensor."""
        b, _, a = x.shape  # batch, channels, anchors
        return self.conv(x.view(b, 4, self.c1, a).transpose(2, 1).softmax(1)).view(b, 4, a)
        # return self.conv(x.view(b, self.c1, 4, a).softmax(1)).view(b, 4, a)


class CBAM(nn.Module):
    def __init__(self, chn_in, chn_out, channel_ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(chn_in, chn_out // channel_ratio, 1, bias=False)
        self.relu1 = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(chn_out // channel_ratio, chn_out, 1, bias=False)

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        
    def forward(self, x):
        # channel attention
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        x = x * out.sigmoid_()
        
        # spatial attention
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv1(out)
        x = x * out.sigmoid_()
        
        return x


class SPP(nn.Module):
    # Spatial pyramid pooling layer used in YOLOv3-SPP
    def __init__(self, c1, c2, k=(5, 9, 13)):
        super(SPP, self).__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * (len(k) + 1), c2, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])

    def forward(self, x):
        x = self.cv1(x)
        return self.cv2(torch.cat([x] + [m(x) for m in self.m], 1))


class SPPF(nn.Module):
    """Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher."""

    def __init__(self, c1, c2, k=5):
        """
        Initializes the SPPF layer with given input/output channels and kernel size.

        This module is equivalent to SPP(k=(5, 9, 13)).
        """
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        """Forward pass through Ghost Convolution block."""
        y = [self.cv1(x)]
        y.extend(self.m(y[-1]) for _ in range(3))
        return self.cv2(torch.cat(y, 1))


class ASPP(nn.Module):
    # Atrous spatial pyramid pooling layer
    def __init__(self, c1, c2, d=(1, 2, 4, 6)):
        super(ASPP, self).__init__()
        assert c1 == c2 and c2 % len(d) == 0
        c_ = c2 // len(d)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.m = nn.ModuleList([Conv(c_, c_, k=3, s=1, p=x, d=x) for x in d])

    def forward(self, x):
        x = self.cv1(x)
        return torch.cat([m(x) for m in self.m], 1)


class Focus(nn.Module):
    # Focus wh information into c-space
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Focus, self).__init__()
        self.conv = Conv(c1 * 4, c2, k, s, p, g, act)
        # self.contract = Contract(gain=2)

    def forward(self, x):  # x(b,c,w,h) -> y(b,4c,w/2,h/2)
        return self.conv(torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1))
        # return self.conv(self.contract(x))


class Blur(nn.Module):
    # Blur c information into wh-space
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Blur, self).__init__()
        self.conv = Conv(c1 // 4, c2, k, s, p, g, act)

    def forward(self, x):  # x(b,4c,w,h) -> y(b,c,2w,2h)
        return self.conv(F.pixel_shuffle(x, 2))


class Contract(nn.Module):
    # Contract width-height into channels, i.e. x(1,64,80,80) to x(1,256,40,40)
    def __init__(self, gain=2):
        super().__init__()
        self.gain = gain

    def forward(self, x):
        N, C, H, W = x.size()  # assert (H / s == 0) and (W / s == 0), 'Indivisible gain'
        s = self.gain
        x = x.view(N, C, H // s, s, W // s, s)  # x(1,64,40,2,40,2)
        x = x.permute(0, 3, 5, 1, 2, 4).contiguous()  # x(1,2,2,64,40,40)
        return x.view(N, C * s * s, H // s, W // s)  # x(1,256,40,40)


class Expand(nn.Module):
    # Expand channels into width-height, i.e. x(1,64,80,80) to x(1,16,160,160)
    def __init__(self, gain=2):
        super().__init__()
        self.gain = gain

    def forward(self, x):
        N, C, H, W = x.size()  # assert C / s ** 2 == 0, 'Indivisible gain'
        s = self.gain
        x = x.view(N, s, s, C // s ** 2, H, W)  # x(1,2,2,16,80,80)
        x = x.permute(0, 3, 4, 1, 5, 2).contiguous()  # x(1,16,80,2,80,2)
        return x.view(N, C // s ** 2, H * s, W * s)  # x(1,16,160,160)


class Concat(nn.Module):
    # Concatenate a list of tensors along dimension
    def __init__(self, dimension=1):
        super(Concat, self).__init__()
        self.d = dimension

    def forward(self, x):
        return torch.cat(x, self.d)


class Add(nn.Module):
    # add a list of tensors
    def __init__(self):
        super(Add, self).__init__()

    def forward(self, x):
        return sum(x)


class Indexer(nn.Module):
    # index a tensor from an input list
    def __init__(self, idx):
        super(Indexer, self).__init__()
        self.idx = idx

    def forward(self, x):
        return x[self.idx]


class Token2Image(nn.Module):
    # convert image tokens back into image shapes with scales
    def __init__(self, scale=1.):
        super(Token2Image, self).__init__()
        self.scale = scale

    def forward(self, x):
        x, (H, W) = x
        # x: torch.Tensor
        B, L, C = x.shape
        h, w = int(round(int(H) * self.scale)), int(round(int(W) * self.scale))
        assert L == h * w
        x = x.view(B, h, w, C).permute(0, 3, 1, 2).contiguous()
        return x


class NMS(nn.Module):
    # Non-Maximum Suppression (NMS) module
    conf = 0.25  # confidence threshold
    iou = 0.45  # IoU threshold
    classes = None  # (optional list) filter by class
    max_det = 1000  # maximum number of detections per image

    def __init__(self):
        super(NMS, self).__init__()

    def forward(self, x):
        return non_max_suppression(x[0], self.conf, iou_thres=self.iou, classes=self.classes, max_det=self.max_det)


class AutoShape(nn.Module):
    # input-robust model wrapper for passing cv2/np/PIL/torch inputs. Includes preprocessing, inference and NMS
    conf = 0.25  # NMS confidence threshold
    iou = 0.45  # NMS IoU threshold
    classes = None  # (optional list) filter by class
    max_det = 1000  # maximum number of detections per image

    def __init__(self, model):
        super(AutoShape, self).__init__()
        self.model = model.eval()

    def autoshape(self):
        print('AutoShape already enabled, skipping... ')  # model already converted to model.autoshape()
        return self

    @torch.no_grad()
    def forward(self, imgs, size=640, augment=False, profile=False):
        # Inference from various sources. For height=640, width=1280, RGB images example inputs are:
        #   filename:   imgs = 'data/images/zidane.jpg'
        #   URI:             = 'https://github.com/ultralytics/yolov5/releases/download/v1.0/zidane.jpg'
        #   OpenCV:          = cv2.imread('image.jpg')[:,:,::-1]  # HWC BGR to RGB x(640,1280,3)
        #   PIL:             = Image.open('image.jpg')  # HWC x(640,1280,3)
        #   numpy:           = np.zeros((640,1280,3))  # HWC
        #   torch:           = torch.zeros(16,3,320,640)  # BCHW (scaled to size=640, 0-1 values)
        #   multiple:        = [Image.open('image1.jpg'), Image.open('image2.jpg'), ...]  # list of images

        t = [time_synchronized()]
        p = next(self.model.parameters())  # for device and type
        if isinstance(imgs, torch.Tensor):  # torch
            with amp.autocast(enabled=p.device.type != 'cpu'):
                return self.model(imgs.to(p.device).type_as(p), augment, profile)  # inference

        # Pre-process
        n, imgs = (len(imgs), imgs) if isinstance(imgs, list) else (1, [imgs])  # number of images, list of images
        shape0, shape1, files = [], [], []  # image and inference shapes, filenames
        for i, im in enumerate(imgs):
            f = f'image{i}'  # filename
            if isinstance(im, str):  # filename or uri
                im, f = np.asarray(Image.open(requests.get(im, stream=True).raw if im.startswith('http') else im)), im
            elif isinstance(im, Image.Image):  # PIL Image
                im, f = np.asarray(im), getattr(im, 'filename', f) or f
            files.append(Path(f).with_suffix('.jpg').name)
            if im.shape[0] < 5:  # image in CHW
                im = im.transpose((1, 2, 0))  # reverse dataloader .transpose(2, 0, 1)
            im = im[:, :, :3] if im.ndim == 3 else np.tile(im[:, :, None], 3)  # enforce 3ch input
            s = im.shape[:2]  # HWC
            shape0.append(s)  # image shape
            g = (size / max(s))  # gain
            shape1.append([y * g for y in s])
            imgs[i] = im if im.data.contiguous else np.ascontiguousarray(im)  # update
        shape1 = [make_divisible(x, int(self.stride.max())) for x in np.stack(shape1, 0).max(0)]  # inference shape
        x = [letterbox(im, new_shape=shape1, auto=False)[0] for im in imgs]  # pad
        x = np.stack(x, 0) if n > 1 else x[0][None]  # stack
        x = np.ascontiguousarray(x.transpose((0, 3, 1, 2)))  # BHWC to BCHW
        x = torch.from_numpy(x).to(p.device).type_as(p) / 255.  # uint8 to fp16/32
        t.append(time_synchronized())

        with amp.autocast(enabled=p.device.type != 'cpu'):
            # Inference
            y = self.model(x, augment, profile)[0]  # forward
            t.append(time_synchronized())

            # Post-process
            y = non_max_suppression(y, self.conf, iou_thres=self.iou, classes=self.classes, max_det=self.max_det)  # NMS
            for i in range(n):
                scale_coords(shape1, y[i][:, :4], shape0[i])

            t.append(time_synchronized())
            return Detections(imgs, y, files, t, self.names, x.shape)


class Detections:
    # detections class for YOLOv5 inference results
    def __init__(self, imgs, pred, files, times=None, names=None, shape=None):
        super(Detections, self).__init__()
        d = pred[0].device  # device
        gn = [torch.tensor([*[im.shape[i] for i in [1, 0, 1, 0]], 1., 1.], device=d) for im in imgs]  # normalizations
        self.imgs = imgs  # list of images as numpy arrays
        self.pred = pred  # list of tensors pred[0] = (xyxy, conf, cls)
        self.names = names  # class names
        self.files = files  # image filenames
        self.xyxy = pred  # xyxy pixels
        self.xywh = [xyxy2xywh(x) for x in pred]  # xywh pixels
        self.xyxyn = [x / g for x, g in zip(self.xyxy, gn)]  # xyxy normalized
        self.xywhn = [x / g for x, g in zip(self.xywh, gn)]  # xywh normalized
        self.n = len(self.pred)  # number of images (batch size)
        self.t = tuple((times[i + 1] - times[i]) * 1000 / self.n for i in range(3))  # timestamps (ms)
        self.s = shape  # inference BCHW shape

    def display(self, pprint=False, show=False, save=False, crop=False, render=False, save_dir=Path('')):
        for i, (im, pred) in enumerate(zip(self.imgs, self.pred)):
            str = f'image {i + 1}/{len(self.pred)}: {im.shape[0]}x{im.shape[1]} '
            if pred is not None:
                for c in pred[:, -1].unique():
                    n = (pred[:, -1] == c).sum()  # detections per class
                    str += f"{n} {self.names[int(c)]}{'s' * (n > 1)}, "  # add to string
                if show or save or render or crop:
                    for *box, conf, cls in pred:  # xyxy, confidence, class
                        label = f'{self.names[int(cls)]} {conf:.2f}'
                        if crop:
                            save_one_box(box, im, file=save_dir / 'crops' / self.names[int(cls)] / self.files[i])
                        else:  # all others
                            plot_one_box(box, im, label=label, color=colors(cls))

            im = Image.fromarray(im.astype(np.uint8)) if isinstance(im, np.ndarray) else im  # from np
            if pprint:
                print(str.rstrip(', '))
            if show:
                im.show(self.files[i])  # show
            if save:
                f = self.files[i]
                im.save(save_dir / f)  # save
                print(f"{'Saved' * (i == 0)} {f}", end=',' if i < self.n - 1 else f' to {save_dir}\n')
            if render:
                self.imgs[i] = np.asarray(im)

    def print(self):
        self.display(pprint=True)  # print results
        print(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {tuple(self.s)}' % self.t)

    def show(self):
        self.display(show=True)  # show results

    def save(self, save_dir='runs/hub/exp'):
        save_dir = increment_path(save_dir, exist_ok=save_dir != 'runs/hub/exp', mkdir=True)  # increment save_dir
        self.display(save=True, save_dir=save_dir)  # save results

    def crop(self, save_dir='runs/hub/exp'):
        save_dir = increment_path(save_dir, exist_ok=save_dir != 'runs/hub/exp', mkdir=True)  # increment save_dir
        self.display(crop=True, save_dir=save_dir)  # crop results
        print(f'Saved results to {save_dir}\n')

    def render(self):
        self.display(render=True)  # render results
        return self.imgs

    def pandas(self):
        # return detections as pandas DataFrames, i.e. print(results.pandas().xyxy[0])
        new = copy(self)  # return copy
        ca = 'xmin', 'ymin', 'xmax', 'ymax', 'confidence', 'class', 'name'  # xyxy columns
        cb = 'xcenter', 'ycenter', 'width', 'height', 'confidence', 'class', 'name'  # xywh columns
        for k, c in zip(['xyxy', 'xyxyn', 'xywh', 'xywhn'], [ca, ca, cb, cb]):
            a = [[x[:5] + [int(x[5]), self.names[int(x[5])]] for x in x.tolist()] for x in getattr(self, k)]  # update
            setattr(new, k, [pd.DataFrame(x, columns=c) for x in a])
        return new

    def tolist(self):
        # return a list of Detections objects, i.e. 'for result in results.tolist():'
        x = [Detections([self.imgs[i]], [self.pred[i]], self.names, self.s) for i in range(self.n)]
        for d in x:
            for k in ['imgs', 'pred', 'xyxy', 'xyxyn', 'xywh', 'xywhn']:
                setattr(d, k, getattr(d, k)[0])  # pop out of list
        return x

    def __len__(self):
        return self.n


class Classify(nn.Module):
    # Classification head, i.e. x(b,c1,20,20) to x(b,c2)
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Classify, self).__init__()
        self.aap = nn.AdaptiveAvgPool2d(1)  # to x(b,c1,1,1)
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g)  # to x(b,c2,1,1)
        self.flat = nn.Flatten()

    def forward(self, x):
        z = torch.cat([self.aap(y) for y in (x if isinstance(x, list) else [x])], 1)  # cat if list
        return self.flat(self.conv(z))  # flatten to x(b,c2)


def get_decoupled_heads(ch, nc, na, type='YOLOv6Head'):
    return nn.ModuleList(eval(type)(x, nc, na) for x in ch)


class YOLOXHead(nn.Module):
    # https://github.com/Megvii-BaseDetection/YOLOX/blob/main/yolox/models/yolo_head.py
    def __init__(self, c1, nc, na, w=1.0):
        super(YOLOXHead, self).__init__()
        self.nc = nc
        self.na = na
        c = int(256 * w)
        self.stem = Conv(c1, c, 1)
        self.cls_conv = nn.Sequential(Conv(c, c, 3, 1), Conv(c, c, 3, 1), Conv(c, c, 3, 1))
        self.reg_conv = nn.Sequential(Conv(c, c, 3, 1), Conv(c, c, 3, 1), Conv(c, c, 3, 1))
        # self.cls_conv = nn.Sequential(Conv(c, c, 3, 1), Conv(c, c, 3, 1))
        # self.reg_conv = nn.Sequential(Conv(c, c, 3, 1), Conv(c, c, 3, 1))
        self.cls_pred = nn.Conv2d(c, nc * na, 1)
        self.reg_pred = nn.Conv2d(c, 4 * na, 1)
        self.obj_pred = nn.Conv2d(c, 1 * na, 1)

    def forward(self, x):
        bs, _, ny, nx = x.shape
        stem = self.stem(x)
        cls_feat = self.cls_conv(stem)
        reg_feat = self.reg_conv(stem)
        cls = self.cls_pred(cls_feat).view(bs, self.na, self.nc, ny, nx)
        reg = self.reg_pred(reg_feat).view(bs, self.na, 4, ny, nx)
        obj = self.obj_pred(reg_feat).view(bs, self.na, 1, ny, nx)
        y = torch.cat((reg, obj, cls), 2)
        return y.view(bs, -1, ny, nx)


class MultiYOLOXHead(YOLOXHead):
    def __init__(self, c1, nc, na, w=1.0):
        super(MultiYOLOXHead, self).__init__(c1, nc, na, w)
        c = int(256 * w)
        self.cls_conv = nn.ModuleList([Conv(c, c, 3, 1), Conv(c, c, 3, 1), Conv(c, c, 3, 1)])
        self.reg_conv = nn.ModuleList([Conv(c, c, 3, 1), Conv(c, c, 3, 1), Conv(c, c, 3, 1)])

    def forward(self, x):
        bs, _, ny, nx = x.shape
        x = self.stem(x)
        cls_feat, reg_feat = x, x
        num_layers = np.random.choice(4)
        for cls_conv, reg_conv in zip(self.cls_conv[:num_layers], self.reg_conv[:num_layers]):
            cls_feat = cls_conv(cls_feat)
            reg_feat = reg_conv(reg_feat)
        cls = self.cls_pred(cls_feat).view(bs, self.na, self.nc, ny, nx)
        reg = self.reg_pred(reg_feat).view(bs, self.na, 4, ny, nx)
        obj = self.obj_pred(reg_feat).view(bs, self.na, 1, ny, nx)
        y = torch.cat((reg, obj, cls), 2)
        return y.view(bs, -1, ny, nx)


class YOLOv6Head(YOLOXHead):
    # https://github.com/meituan/YOLOv6/blob/main/yolov6/models/effidehead.py
    def __init__(self, c1, nc, na):
        super(YOLOv6Head, self).__init__(c1, nc, na)
        self.nc = nc
        self.na = na
        c = c1
        self.stem = Conv(c1, c, 1)
        self.cls_conv = Conv(c, c, 3, 1)
        self.reg_conv = Conv(c, c, 3, 1)
        self.cls_pred = nn.Conv2d(c, nc * na, 1)
        self.reg_pred = nn.Conv2d(c, 4 * na, 1)
        self.obj_pred = nn.Conv2d(c, 1 * na, 1)

