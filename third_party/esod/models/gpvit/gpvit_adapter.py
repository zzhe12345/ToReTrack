# Copyright (c) Shanghai AI Lab. All rights reserved.
import logging
import math
from functools import partial
from typing import Optional, List, Tuple
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.models.builder import BACKBONES
from timm.models.layers import DropPath, trunc_normal_
from torch.nn.init import normal_

from mmcls.gpvit_dev.models.backbones.gpvit import GPViT, resize_pos_embed
from .adapter_modules import SpatialPriorModule, InteractionBlock, get_reference_points, MSDeformAttn

_logger = logging.getLogger(__name__)


@BACKBONES.register_module(force=True)
class GPViTAdapter(GPViT):
    def __init__(self,
                 pretrain_size=224,
                 conv_inplane=64,
                 n_points=4,
                 deform_num_heads=6,
                 init_values=0.,
                 interaction_indexes=None,
                 with_cffn=True,
                 cffn_ratio=0.25,
                 deform_ratio=1.0,
                 add_vit_feature=True,
                 use_extra_extractor=True,
                 att_with_cp=False,
                 group_with_cp=False,
                 *args,
                 **kwargs):

        self.att_with_cp = att_with_cp
        self.group_with_cp = group_with_cp

        super().__init__(*args, **kwargs)

        self.num_classes = 80
        self.cls_token = None
        self.num_block = len(self.layers)
        self.pretrain_size = (pretrain_size, pretrain_size)
        self.interaction_indexes = interaction_indexes
        self.add_vit_feature = add_vit_feature
        embed_dim = self.embed_dims

        self.interactions = nn.Sequential(*[
            InteractionBlock_GPViT(
                dim=embed_dim,
                num_heads=deform_num_heads,
                n_points=n_points,
                init_values=init_values,
                drop_path=self.drop_path_rate,
                # norm_layer=self.norm1,
                with_cffn=with_cffn,
                cffn_ratio=cffn_ratio,
                deform_ratio=deform_ratio,
                extra_extractor=((True if i == len(interaction_indexes) - 1 else False) and use_extra_extractor),
                down_stride=8
            )
            for i in range(len(interaction_indexes))
        ])
        self.up = nn.ConvTranspose2d(embed_dim, embed_dim, 2, 2)
        self.ad_norm1 = nn.BatchNorm2d(embed_dim)
        self.ad_norm2 = nn.BatchNorm2d(embed_dim)
        self.ad_norm3 = nn.BatchNorm2d(embed_dim)
        self.ad_norm4 = nn.BatchNorm2d(embed_dim)

        self.up.apply(self._init_weights)
        self.interactions.apply(self._init_weights)
        self.apply(self._init_deform_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm) or isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def _init_deform_weights(self, m):
        if isinstance(m, MSDeformAttn):
            m._reset_parameters()

    def _get_pos_embed(self, pos_embed, H, W):
        pos_embed = pos_embed.reshape(
            1, self.pretrain_size[0] // 16, self.pretrain_size[1] // 16, -1).permute(0, 3, 1, 2)
        pos_embed = F.interpolate(pos_embed, size=(H, W), mode='bicubic', align_corners=False).\
            reshape(1, -1, H * W).permute(0, 2, 1)
        return pos_embed

    def forward(self, x):        
        deform_inputs1, deform_inputs2 = deform_inputs(x)

        # SPM forward
        c1, c2, c3, c4 = self.spm(x)  # s4, s8, s16, s32
        c2, c3, c4 = self._add_level_embed(c2, c3, c4)
        c = torch.cat([c2, c3, c4], dim=1)

        B = x.shape[0]
        x, patch_resolution = self.patch_embed(x)
        H, W = patch_resolution
        bs, n, dim = x.shape
        pos_embed = resize_pos_embed(
            self.pos_embed,
            self.patch_resolution,
            patch_resolution,
            mode=self.interpolate_mode,
            num_extra_tokens=0)

        x = x + pos_embed
        x = self.drop_after_pos(x)

        # Interaction
        for i, layer in enumerate(self.interactions):
            indexes = self.interaction_indexes[i]
            x, c = layer(x, c, self.layers[indexes[0]:indexes[-1] + 1],
                         deform_inputs1, deform_inputs2, patch_resolution)

        # Split & Reshape
        c2 = c[:, 0:c2.size(1), :]
        c3 = c[:, c2.size(1):c2.size(1) + c3.size(1), :]
        c4 = c[:, c2.size(1) + c3.size(1):, :]

        c2 = c2.transpose(1, 2).view(bs, dim, H, W).contiguous()
        c3 = c3.transpose(1, 2).view(bs, dim, H // 2, W // 2).contiguous()
        c4 = c4.transpose(1, 2).view(bs, dim, H // 4, W // 4).contiguous()
        c1 = self.up(c2) + c1

        if self.add_vit_feature:
            x2 = x.transpose(1, 2).view(bs, dim, H, W).contiguous()
            x1 = F.interpolate(x2, scale_factor=2, mode='bilinear', align_corners=False)
            x3 = F.interpolate(x2, scale_factor=0.5, mode='bilinear', align_corners=False)
            x4 = F.interpolate(x2, scale_factor=0.25, mode='bilinear', align_corners=False)
            c1, c2, c3, c4 = c1 + x1, c2 + x2, c3 + x3, c4 + x4

        # Final Norm
        f1 = self.ad_norm1(c1)
        f2 = self.ad_norm2(c2)
        f3 = self.ad_norm3(c3)
        f4 = self.ad_norm4(c4)
        return [f1, f2, f3, f4]


@BACKBONES.register_module(force=True)
class GPViTAdapterSingleStageESOD(GPViTAdapter):
    def __init__(self,
                 arch="L2",
                 pretrain_size=224,
                 conv_inplane=64,
                 n_points=4,
                 deform_num_heads=6,
                 init_values=0.,
                 interaction_indexes=[[0, 2], [3, 5], [6, 8], [9, 11]],  ## update
                 with_cffn=True,
                 cffn_ratio=0.25,
                 deform_ratio=1.0,
                 add_vit_feature=True,
                 use_extra_extractor=True,
                 att_with_cp=False,
                 group_with_cp=False,
                 *args,
                 **kwargs):
        self.att_with_cp = att_with_cp
        self.group_with_cp = group_with_cp

        ## update
        kwargs.update({'arch': arch, "drop_path_rate": 0.1, "out_indices": (11,), 
                       "final_norm": False, "convert_syncbn": False})

        super(GPViTAdapter, self).__init__(*args, **kwargs)

        self.num_classes = 80
        self.cls_token = None
        self.num_block = len(self.layers)
        self.pretrain_size = (pretrain_size, pretrain_size)
        self.interaction_indexes = interaction_indexes
        self.add_vit_feature = add_vit_feature
        embed_dim = self.embed_dims

        self.interactions = nn.Sequential(*[
            InteractionBlock_GPViT(
                dim=embed_dim,
                num_heads=deform_num_heads,
                n_points=n_points,
                init_values=init_values,
                drop_path=self.drop_path_rate,
                # norm_layer=self.norm1,
                with_cffn=with_cffn,
                cffn_ratio=cffn_ratio,
                deform_ratio=deform_ratio,
                extra_extractor=((True if i == len(interaction_indexes) - 1 else False) and use_extra_extractor),
                down_stride=8
            )
            for i in range(len(interaction_indexes))
        ])
        self.ad_norm2 = nn.BatchNorm2d(embed_dim)
        self.ad_norm3 = nn.BatchNorm2d(embed_dim)
        self.ad_norm4 = nn.BatchNorm2d(embed_dim)

        self.interactions.apply(self._init_weights)
        self.apply(self._init_deform_weights)
    
    def build_mask_indices(self, clusters: Optional[torch.Tensor], feat_size: Tuple[int, int]):
        if clusters is None:
            return None, None
        assert clusters.size(0) > 0

        H, W = feat_size
        device = clusters.device

        bi, x1, y1, x2, y2 = clusters.chunk(5, dim=1)  # shape(n,1)
        w, h = int(x2[0, 0] - x1[0, 0]), int(y2[0, 0] - y1[0, 0])
        if getattr(self, 'slice_ratio', None) is None:
            self.slice_ratio = (H // h) #.item()
        assert H // h == W // w == self.slice_ratio
        if getattr(self, 'grid_off', None) is None or self.grid_off.size(1) != w*h:
            gy, gx = torch.meshgrid(torch.arange(h), torch.arange(w))            
            gxy = torch.stack((gy.reshape(-1), gx.reshape(-1)), dim=0)  # shape(2, w*h)
            self.grid_off = gxy.to(device)
        gy, gx = self.grid_off.chunk(2, dim=0)  # shape(1,w*h)
        mask_indices = (bi * H * W + (gy + y1) * W + (gx + x1)).view(-1)  # shape(n * w*h)

        return mask_indices, self.slice_ratio

    def feat_slice(self, featmaps: List[torch.Tensor], clusters: torch.Tensor, scales: List[int]):
        assert len(featmaps) == len(scales)
        featmaps_new = []

        B, C, H, W = featmaps[0].shape
        device = featmaps[0].device

        bi, x1, y1, x2, y2 = clusters.chunk(5, dim=1)  # shape(n,1)
        w, h = (x2[0, 0] - x1[0, 0]).item(), (y2[0, 0] - y1[0, 0]).item()
        assert H // h == W // w == self.slice_ratio
        if getattr(self, 'fs_grids', None) is None or self.fs_grids[0].size(1) != w*h:
            self.fs_grids = []
            for s in scales:
                gy, gx = torch.meshgrid(torch.arange(h//s), torch.arange(w//s))
                gxy = torch.stack((gy.reshape(-1), gx.reshape(-1)), dim=0)  # shape(2, w*h)
                self.fs_grids.append(gxy.to(device))

        for fi, s in enumerate(scales):
            t, l = y1 // s, x1 // s
            gj, gi = self.fs_grids[fi].chunk(2, dim=0)  # shape(1,w*h)
            H_, W_ = H // s, W // s
            mask_indices = (bi * H_ * W_ + (gj + t) * W_ + (gi + l)).view(-1)  # shape(n * w*h)

            fm = featmaps[fi].flatten(2).transpose(1, 2).contiguous()  # (B,C,H,W) -> (B,H*W,C)
            fm = fm.view(-1, C)[mask_indices].view(-1, h//s, w//s, C)
            fm = fm.permute(0, 3, 1, 2).contiguous()
            featmaps_new.append(fm)
        
        return featmaps_new

    def feat_slice2(self, featembs: List[torch.Tensor], scales: List[int],
                    clusters: torch.Tensor, mask_patch_resolution: List[int]):
        assert len(featembs) == len(scales)
        h, w = mask_patch_resolution  # shape of feature patch
        H, W = h * self.slice_ratio, w * self.slice_ratio  # shape of feature map
        device = featembs[0].device
        featembs_new = []

        if getattr(self, 'fs_grids', None) is None or self.fs_grids[0].size(1) != h*w:
            self.fs_grids = []
            for s in scales:
                gy, gx = torch.meshgrid(torch.arange(h//s), torch.arange(w//s))
                gxy = torch.stack((gy.reshape(-1), gx.reshape(-1)), dim=0)  # shape(2, w*h)
                self.fs_grids.append(gxy.to(device))

        bi, x1, y1, x2, y2 = clusters.chunk(5, dim=1)  # shape(n,1)
        for fi, (fm, s) in enumerate(zip(featembs, scales)):
            B, L, C = fm.shape
            fm = fm.view(-1, C)

            t, l = y1 // s, x1 // s
            gj, gi = self.fs_grids[fi].chunk(2, dim=0)  # shape(1,w*h)
            H_, W_ = H // s, W // s
            mask_indices = (bi * H_ * W_ + (gj + t) * W_ + (gi + l)).view(-1)  # shape(n * w*h)

            fm = fm[mask_indices].contiguous().view(-1, (h//s)*(w//s), C)
            featembs_new.append(fm)

        return featembs_new

    def forward(self, x):
        assert isinstance(x, list), type(x)
        if len(x) == 2:
            x, (c2, c3, c4) = x
            clusters = None
        else:
            # mask: tensor(bool), shape(bs,h//8,w//8)
            x, (c2, c3, c4), clusters = x
        
        # 双向Deformable Attention
        deform_inputs1, deform_inputs2 = deform_inputs(x)
        
        B, C, H, W = x.shape
        mask_indices, cluster_size_ratio = self.build_mask_indices(clusters, (H//8, W//8))

        B = x.shape[0]
        x, patch_resolution = self.patch_embed(x)  # 8倍下采样
        # x: shape(1, h/8*w/8, ndim), serve as query
        assert tuple(patch_resolution) == (H // 8, W // 8)
        H, W = patch_resolution
        bs, n, dim = x.shape
        pos_embed = resize_pos_embed(
            self.pos_embed,
            self.patch_resolution,
            patch_resolution,
            mode=self.interpolate_mode,
            num_extra_tokens=0)

        x = x + pos_embed
        x = self.drop_after_pos(x)

        if mask_indices is not None and True:
            # update (slice) features and indices
            assert cluster_size_ratio == 8
            deform_inputs1, deform_inputs2 = \
                deform_inputs(torch.zeros((0, 0, H, W), device=x.device))
            patch_resolution = (H // cluster_size_ratio, W // cluster_size_ratio)
            H, W = patch_resolution
            x, c2, c3, c4 = self.feat_slice2([x, c2, c3, c4], [1, 1, 2, 4], clusters, patch_resolution)
            bs = x.size(0)
            mask_indices, cluster_size_ratio = None, None

        # SPM forward，独立的特征金字塔，下采样率为8/16/32
        # c: shape(bs, h/8*w/8 + h/16*w/16 + h/32*w/32, ndim), serve as feature
        c = torch.cat([c2, c3, c4], dim=1)

        # Interaction
        for i, layer in enumerate(self.interactions):
            indexes = self.interaction_indexes[i]
            x, c = layer(x, c, self.layers[indexes[0]:indexes[-1] + 1],
                         deform_inputs1, deform_inputs2, patch_resolution, 
                         mask_indices=mask_indices, cluster_size_ratio=cluster_size_ratio)

        # Split & Reshape
        c2 = c[:, 0:c2.size(1), :]
        c3 = c[:, c2.size(1):c2.size(1) + c3.size(1), :]
        c4 = c[:, c2.size(1) + c3.size(1):, :]

        c2 = c2.transpose(1, 2).view(bs, dim, H, W).contiguous()
        c3 = c3.transpose(1, 2).view(bs, dim, H // 2, W // 2).contiguous()
        c4 = c4.transpose(1, 2).view(bs, dim, H // 4, W // 4).contiguous()

        if self.add_vit_feature:
            x2 = x.transpose(1, 2).view(bs, dim, H, W).contiguous()
            x3 = F.interpolate(x2, scale_factor=0.5, mode='bilinear', align_corners=False)
            x4 = F.interpolate(x2, scale_factor=0.25, mode='bilinear', align_corners=False)
            c2, c3, c4 = c2 + x2, c3 + x3, c4 + x4

        # Final Norm
        f2 = self.ad_norm2(c2)
        f3 = self.ad_norm3(c3)
        f4 = self.ad_norm4(c4)

        # torch.cuda.synchronize()
        # t0 = time.time()
        if mask_indices is not None:
            f2, f3, f4 = self.feat_slice([f2, f3, f4], clusters, [1, 2, 4])
        # torch.cuda.synchronize()
        # t1 = time.time()
        # print(f"Feature slicing cost {(t1-t0)*1000:.2f}ms")  # 7ms

        return [f2, f3, f4]

    def forward_org(self, x):
        # x: shape(1, 3, 768, 1344)

        # 双向Deformable Attention参数
        deform_inputs1, deform_inputs2 = deform_inputs(x)

        # SPM forward，独立的特征金字塔，下采样率为8/16/32
        c2, c3, c4 = self.spm(x)  # s4, s8, s16, s32
        c2, c3, c4 = self._add_level_embed(c2, c3, c4)
        # c: shape(bs, h/8*w/8 + h/16*w/16 + h/32*w/32, ndim), serve as feature
        c = torch.cat([c2, c3, c4], dim=1)

        B = x.shape[0]
        x, patch_resolution = self.patch_embed(x)  # 8倍下采样
        # x: shape(1, h/8*w/8, ndim), serve as query
        H, W = patch_resolution
        bs, n, dim = x.shape
        pos_embed = resize_pos_embed(
            self.pos_embed,
            self.patch_resolution,
            patch_resolution,
            mode=self.interpolate_mode,
            num_extra_tokens=0)

        x = x + pos_embed
        x = self.drop_after_pos(x)

        # Interaction
        for i, layer in enumerate(self.interactions):
            indexes = self.interaction_indexes[i]
            x, c = layer(x, c, self.layers[indexes[0]:indexes[-1] + 1],
                         deform_inputs1, deform_inputs2, patch_resolution)

        # Split & Reshape
        c2 = c[:, 0:c2.size(1), :]
        c3 = c[:, c2.size(1):c2.size(1) + c3.size(1), :]
        c4 = c[:, c2.size(1) + c3.size(1):, :]

        c2 = c2.transpose(1, 2).view(bs, dim, H, W).contiguous()
        c3 = c3.transpose(1, 2).view(bs, dim, H // 2, W // 2).contiguous()
        c4 = c4.transpose(1, 2).view(bs, dim, H // 4, W // 4).contiguous()

        if self.add_vit_feature:
            x2 = x.transpose(1, 2).view(bs, dim, H, W).contiguous()
            x3 = F.interpolate(x2, scale_factor=0.5, mode='bilinear', align_corners=False)
            x4 = F.interpolate(x2, scale_factor=0.25, mode='bilinear', align_corners=False)
            c2, c3, c4 = c2 + x2, c3 + x3, c4 + x4

        # Final Norm
        f2 = self.ad_norm2(c2)
        f3 = self.ad_norm3(c3)
        f4 = self.ad_norm4(c4)
        return [f2, f3, f4]

class InteractionBlock_GPViT(InteractionBlock):
    @staticmethod
    def chunk_feat(x: torch.Tensor, mask_indices, patch_resolution, cluster_size_ratio=8):
        if mask_indices is None:
            return x, patch_resolution
        B, L, C = x.shape
        ES = cluster_size_ratio
        assert L % (ES*ES) == 0 and len(mask_indices) % (L//(ES*ES)) == 0
        z = x.view(-1, C)[mask_indices].view(-1, L//(ES*ES), C).contiguous()

        H, W = patch_resolution
        return z, (H//ES, W//ES)
    
    @staticmethod
    def recover_feat(x: torch.Tensor, mask_indices, x0):
        if mask_indices is None:
            return x
        B, L, C = x0.shape
        x0.view(-1, C)[mask_indices] = x.view(-1, C).type_as(x0).contiguous()
        return x0

    def forward(self, x, c, blocks, deform_inputs1, deform_inputs2, patch_resolution, 
                mask_indices=None, cluster_size_ratio=None):
        H, W = patch_resolution
        x = x.contiguous()

        COUNT_LATENCY = False
        if COUNT_LATENCY:
            timestamps = []
            torch.cuda.synchronize()
            timestamps.append(time.time())
        x = self.injector(query=x,
                          reference_points=deform_inputs1[0],
                          feat=c,
                          spatial_shapes=deform_inputs1[1],
                          level_start_index=deform_inputs1[2])
        if COUNT_LATENCY:
            torch.cuda.synchronize()
            timestamps.append(time.time())
        
        x0 = x
        x, patch_resolution = \
            self.chunk_feat(x, mask_indices, patch_resolution, cluster_size_ratio)
        if COUNT_LATENCY:
            torch.cuda.synchronize()
            timestamps.append(time.time())

        if x.size(0) > 0:
            for idx, blk in enumerate(blocks):
                x = blk(x, patch_resolution)
        if COUNT_LATENCY:
            torch.cuda.synchronize()
            timestamps.append(time.time())

        x = self.recover_feat(x, mask_indices, x0)
        if COUNT_LATENCY:
            torch.cuda.synchronize()
            timestamps.append(time.time())

        c = self.extractor(query=c,
                           reference_points=deform_inputs2[0],
                           feat=x,
                           spatial_shapes=deform_inputs2[1],
                           level_start_index=deform_inputs2[2],
                           H=H, W=W)
        if self.extra_extractors is not None:
            for extractor in self.extra_extractors:
                c = extractor(query=c,
                              reference_points=deform_inputs2[0],
                              feat=x,
                              spatial_shapes=deform_inputs2[1],
                              level_start_index=deform_inputs2[2],
                              H=H, W=W)
        if COUNT_LATENCY:
            torch.cuda.synchronize()
            timestamps.append(time.time())
            
            s = ''
            for i in range(len(timestamps) - 1):
                s += f'{(timestamps[i+1] - timestamps[i])*1000:.2f}ms\t'
            print(s)

        return x, c


def deform_inputs(x):
    bs, c, h, w = x.shape
    spatial_shapes = torch.as_tensor([(h // 8, w // 8),
                                      (h // 16, w // 16),
                                      (h // 32, w // 32)],
                                     dtype=torch.long, device=x.device)
    # tensor([0, h/8*w/8, h/8*w/8 + h/16*w/16])
    level_start_index = torch.cat((
        spatial_shapes.new_zeros((1,)), 
        spatial_shapes.prod(1).cumsum(0)[:-1]
    ))
    # shape(1, h/8*w/8, 1, 2), (xi, yi)从0.5/(h/8)开始，步长为1.0/(h/8)
    reference_points = get_reference_points([(h // 8, w // 8)], x.device)
    deform_inputs1 = [reference_points, spatial_shapes, level_start_index]

    spatial_shapes = torch.as_tensor([(h // 8, w // 8)], dtype=torch.long, device=x.device)
    # tensor([0])
    level_start_index = torch.cat((
        spatial_shapes.new_zeros((1,)), 
        spatial_shapes.prod(1).cumsum(0)[:-1]
    ))
    # shape(1, h/8*w/8+ h/16*w/16+ h/32*w/32, 1, 2), (xi, yi) = range(0.5/(h/s), 1.0, 1.0/(h/s))，s可变
    reference_points = get_reference_points([(h // 8, w // 8),
                                             (h // 16, w // 16),
                                             (h // 32, w // 32)], x.device)
    deform_inputs2 = [reference_points, spatial_shapes, level_start_index]
    return deform_inputs1, deform_inputs2
