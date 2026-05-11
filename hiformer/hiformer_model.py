# -*- coding: utf-8 -*-
"""
HiFormer — Hierarchical Multi-scale Representations using Transformers for
Medical Image Segmentation.

Architecture overview:
  - PyramidFeatures: dual-branch encoder combining a pretrained ResNet CNN
    with a pretrained Swin Transformer, fusing features at three pyramid levels.
  - All2Cross:  multi-scale cross-attention blocks exchange information between
    the two resolution streams produced by PyramidFeatures.
  - ConvUpsample + SegmentationHead: lightweight convolutional decoder that
    upsamples and combines the two streams into the final segmentation map.

Dependencies:
    pip install torch torchvision timm einops

Pretrained Swin-T weights (swin_tiny_patch4_window7_224.pth) must be
downloaded separately and the path supplied via config.swin_pretrained_path.
Download from: https://github.com/microsoft/Swin-Transformer

Usage:
    from hiformer.hiformer_model import HiFormer, get_hiformer_config
    config = get_hiformer_config()
    model  = HiFormer(config=config, img_size=224, in_chans=3, n_classes=1)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from timm.models.layers import trunc_normal_
from einops import rearrange
from einops.layers.torch import Rearrange

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─────────────────────────────────────────────────────────────
# Config helper
# ─────────────────────────────────────────────────────────────

class HiFormerConfig:
    """Default HiFormer-B configuration for 224×224 input."""
    swin_pretrained_path = 'pretrained/swin_tiny_patch4_window7_224.pth'
    cnn_backbone         = 'resnet50'
    resnet_pretrained    = True
    image_size           = 224
    patch_size           = 4
    # CNN feature-map channels at each pyramid level
    cnn_pyramid_fm  = [256, 512, 1024]
    # Swin projected channels at each pyramid level
    swin_pyramid_fm = [96, 192, 384]
    # Cross-attention block depths: (local_depth, local_depth, cross_depth)
    depth           = [(2, 2, 2), (2, 2, 2), (2, 2, 2)]
    num_heads       = [3, 6, 12]
    mlp_ratio       = 4.0
    qkv_bias        = True
    qk_scale        = None
    drop_rate       = 0.0
    attn_drop_rate  = 0.0
    drop_path_rate  = 0.1
    cross_pos_embed  = True


def get_hiformer_config():
    return HiFormerConfig()


# ─────────────────────────────────────────────────────────────
# Swin-Transformer primitives  (minimal self-contained impl.)
# ─────────────────────────────────────────────────────────────

def window_partition(x, window_size):
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)


def window_reverse(windows, window_size, H, W):
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)


class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads, qkv_bias=True,
                 qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim         = dim
        self.window_size = window_size   # (Wh, Ww)
        self.num_heads   = num_heads
        head_dim         = dim // num_heads
        self.scale       = qk_scale or head_dim ** -0.5

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))
        trunc_normal_(self.relative_position_bias_table, std=.02)

        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords   = torch.stack(torch.meshgrid([coords_h, coords_w]))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        self.register_buffer("relative_position_index",
                             relative_coords.sum(-1))

        self.qkv      = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj      = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax   = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads,
                                   C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q   = q * self.scale
        attn = q @ k.transpose(-2, -1)

        rpb  = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(self.window_size[0] * self.window_size[1],
               self.window_size[0] * self.window_size[1], -1)
        attn = attn + rpb.permute(2, 0, 1).unsqueeze(0)

        if mask is not None:
            attn = attn.view(B_ // mask.shape[0], mask.shape[0],
                             self.num_heads, N, N)
            attn = attn + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        return self.proj_drop(self.proj(x))


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.):
        super().__init__()
        hidden_features = hidden_features or in_features
        out_features    = out_features    or in_features
        self.fc1  = nn.Linear(in_features, hidden_features)
        self.act  = nn.GELU()
        self.fc2  = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        return self.drop(self.fc2(self.drop(self.act(self.fc1(x)))))


class SwinTransformerBlock(nn.Module):
    def __init__(self, dim, input_resolution, num_heads, window_size=7,
                 shift_size=0, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop=0., attn_drop=0., drop_path=0., norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim              = dim
        self.input_resolution = input_resolution
        self.num_heads        = num_heads
        self.window_size      = window_size
        self.shift_size       = shift_size
        self.mlp_ratio        = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            self.shift_size  = 0
            self.window_size = min(self.input_resolution)

        self.norm1 = norm_layer(dim)
        self.attn  = WindowAttention(
            dim, window_size=(self.window_size, self.window_size),
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = nn.Identity()   # simplified — use timm's DropPath for full impl.
        self.norm2     = norm_layer(dim)
        self.mlp       = Mlp(in_features=dim,
                             hidden_features=int(dim * mlp_ratio), drop=drop)

        if self.shift_size > 0:
            H, W  = self.input_resolution
            img_mask = torch.zeros(1, H, W, 1)
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1
            mask_windows = window_partition(img_mask, self.window_size)
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask    = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask    = attn_mask.masked_fill(attn_mask != 0,
                                                 float(-100.0)).masked_fill(attn_mask == 0, 0.0)
        else:
            attn_mask = None
        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x):
        H, W  = self.input_resolution
        B, L, C = x.shape
        shortcut = x
        x = self.norm1(x).view(B, H, W, C)
        if self.shift_size > 0:
            x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        x_windows = window_partition(x, self.window_size).view(
            -1, self.window_size * self.window_size, C)
        attn_out  = self.attn(x_windows, mask=self.attn_mask)
        x = window_reverse(attn_out.view(-1, self.window_size, self.window_size, C),
                           self.window_size, H, W)
        if self.shift_size > 0:
            x = torch.roll(x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        x = shortcut + self.drop_path(x.view(B, H * W, C))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class BasicLayer(nn.Module):
    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0.,
                 attn_drop=0., drop_path=0., norm_layer=nn.LayerNorm, downsample=None):
        super().__init__()
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=dim, input_resolution=input_resolution, num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop, attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer)
            for i in range(depth)
        ])
        self.downsample = downsample

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x


class PatchMerging(nn.Module):
    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim       = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm      = norm_layer(4 * dim)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        x = x.view(B, H, W, C)
        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x  = self.reduction(self.norm(torch.cat([x0, x1, x2, x3], -1)))
        return x.view(B, -1, 2 * C)


# ─────────────────────────────────────────────────────────────
# Cross-attention block
# ─────────────────────────────────────────────────────────────

class CrossAttentionBlock(nn.Module):
    """Single cross-attention layer between two feature streams."""
    def __init__(self, dim_q, dim_kv, num_heads=8, dropout=0.0):
        super().__init__()
        self.num_heads = num_heads
        self.scale     = (dim_q // num_heads) ** -0.5
        self.norm_q    = nn.LayerNorm(dim_q)
        self.norm_kv   = nn.LayerNorm(dim_kv)
        self.q         = nn.Linear(dim_q,  dim_q,  bias=False)
        self.k         = nn.Linear(dim_kv, dim_q,  bias=False)
        self.v         = nn.Linear(dim_kv, dim_q,  bias=False)
        self.proj      = nn.Linear(dim_q,  dim_q)
        self.drop      = nn.Dropout(dropout)
        self.norm_out  = nn.LayerNorm(dim_q)
        self.mlp       = Mlp(dim_q, int(dim_q * 4))

    def forward(self, x, context):
        B, N, C = x.shape
        q = self.q(self.norm_q(x)).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.k(self.norm_kv(context)).reshape(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.v(self.norm_kv(context)).reshape(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        attn = self.drop(F.softmax(q @ k.transpose(-2, -1) * self.scale, dim=-1))
        out  = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x    = x + self.proj(out)
        x    = x + self.mlp(self.norm_out(x))
        return x


class MultiScaleBlock(nn.Module):
    """Process both streams with local self-attention then cross-attend."""
    def __init__(self, embed_dim, num_patches, block_config, num_heads,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop=0., attn_drop=0., drop_path=None, norm_layer=nn.LayerNorm):
        super().__init__()
        local_depth_0, local_depth_1, cross_depth = block_config
        d0, d1 = embed_dim

        # Local self-attention for stream 0
        self.local_0 = nn.ModuleList([
            SwinTransformerBlock(
                dim=d0,
                input_resolution=(int(num_patches[0] ** 0.5), int(num_patches[0] ** 0.5)),
                num_heads=num_heads[0], window_size=7,
                shift_size=0 if i % 2 == 0 else 3,
                mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                drop=drop, attn_drop=attn_drop, norm_layer=norm_layer)
            for i in range(local_depth_0)
        ])

        # Local self-attention for stream 1
        self.local_1 = nn.ModuleList([
            SwinTransformerBlock(
                dim=d1,
                input_resolution=(int(num_patches[1] ** 0.5), int(num_patches[1] ** 0.5)),
                num_heads=num_heads[1], window_size=7,
                shift_size=0 if i % 2 == 0 else 3,
                mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                drop=drop, attn_drop=attn_drop, norm_layer=norm_layer)
            for i in range(local_depth_1)
        ])

        # Cross-attention blocks
        self.cross_01 = nn.ModuleList([
            CrossAttentionBlock(d0, d1, num_heads=num_heads[0]) for _ in range(cross_depth)
        ])
        self.cross_10 = nn.ModuleList([
            CrossAttentionBlock(d1, d0, num_heads=num_heads[1]) for _ in range(cross_depth)
        ])

    def forward(self, xs):
        x0, x1 = xs
        for blk in self.local_0:
            x0 = blk(x0)
        for blk in self.local_1:
            x1 = blk(x1)
        for ca01, ca10 in zip(self.cross_01, self.cross_10):
            x0_new = ca01(x0, x1)
            x1_new = ca10(x1, x0)
            x0, x1 = x0_new, x1_new
        return [x0, x1]


# ─────────────────────────────────────────────────────────────
# Pyramid feature extractor
# ─────────────────────────────────────────────────────────────

class SwinTransformerEncoder(nn.Module):
    """Minimal Swin Transformer used as encoder (3 stages, no patch embed)."""
    def __init__(self, img_size=224, embed_dim=96,
                 depths=(2, 2, 6), num_heads=(3, 6, 12),
                 window_size=7, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, patch_size=4):
        super().__init__()
        patches_resolution = [img_size // patch_size, img_size // patch_size]
        self.num_layers    = len(depths)
        self.embed_dim     = embed_dim
        self.pos_drop      = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(
                dim=int(embed_dim * 2 ** i_layer),
                input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                  patches_resolution[1] // (2 ** i_layer)),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                downsample=None,
            )
            self.layers.append(layer)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


class PyramidFeatures(nn.Module):
    def __init__(self, config, img_size=224, in_channels=3):
        super().__init__()
        self.swin_transformer = SwinTransformerEncoder(
            img_size=img_size,
            embed_dim=config.swin_pyramid_fm[0],
            depths=[2, 2, 6],
            num_heads=[3, 6, 12],
            window_size=7,
            patch_size=config.patch_size,
        )

        # Load pretrained Swin weights (gracefully skip if not found)
        model_path = config.swin_pretrained_path
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=device)
            state = checkpoint.get('model', checkpoint)
            # Filter keys that match our encoder
            filtered = {k: v for k, v in state.items()
                        if k.startswith('layers') and 'downsample' not in k
                        and 'layers.3' not in k}
            missing, unexpected = self.swin_transformer.load_state_dict(filtered, strict=False)
            print(f'[HiFormer] Loaded Swin weights. Missing: {len(missing)}, Unexpected: {len(unexpected)}')
        else:
            print(f'[HiFormer] WARNING: pretrained Swin weights not found at {model_path}. '
                  'Training from scratch.')

        resnet  = eval(f"torchvision.models.{config.cnn_backbone}(pretrained={config.resnet_pretrained})")
        resnet_children = list(resnet.children())
        self.resnet_early = nn.Sequential(*resnet_children[:5])  # → 256ch stride-4
        self.resnet_mid   = resnet_children[5]                    # → 512ch stride-8
        self.resnet_deep  = resnet_children[6]                    # → 1024ch stride-16

        self.p1_ch  = nn.Conv2d(config.cnn_pyramid_fm[0], config.swin_pyramid_fm[0], 1)
        self.p1_pm  = PatchMerging(
            (img_size // config.patch_size, img_size // config.patch_size),
            config.swin_pyramid_fm[0])
        self.norm_1 = nn.LayerNorm(config.swin_pyramid_fm[0])
        self.avg_1  = nn.AdaptiveAvgPool1d(1)

        self.p2_ch  = nn.Conv2d(config.cnn_pyramid_fm[1], config.swin_pyramid_fm[1], 1)
        self.p2_pm  = PatchMerging(
            (img_size // config.patch_size // 2, img_size // config.patch_size // 2),
            config.swin_pyramid_fm[1])

        self.p3_ch  = nn.Conv2d(config.cnn_pyramid_fm[2], config.swin_pyramid_fm[2], 1)
        self.norm_2 = nn.LayerNorm(config.swin_pyramid_fm[2])
        self.avg_2  = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        fm1 = self.resnet_early(x)                            # (B,256,H/4,W/4)
        fm1_ch = self.p1_ch(fm1)
        fm1_seq = Rearrange('b c h w -> b (h w) c')(fm1_ch)

        sw1        = self.swin_transformer.layers[0](fm1_seq)
        sw1_skip   = fm1_seq + sw1
        norm1      = self.norm_1(sw1_skip)
        cls1       = Rearrange('b c 1 -> b 1 c')(self.avg_1(norm1.transpose(1, 2)))
        fm1_merged = self.p1_pm(sw1_skip)

        fm2        = self.resnet_mid(fm1)
        fm2_ch     = self.p2_ch(fm2)
        fm2_seq    = Rearrange('b c h w -> b (h w) c')(fm2_ch)
        sw2        = self.swin_transformer.layers[1](fm1_merged)
        fm2_skip   = fm2_seq + sw2
        fm2_merged = self.p2_pm(fm2_skip)

        fm3        = self.resnet_deep(fm2)
        fm3_ch     = self.p3_ch(fm3)
        fm3_seq    = Rearrange('b c h w -> b (h w) c')(fm3_ch)
        sw3        = self.swin_transformer.layers[2](fm2_merged)
        fm3_skip   = fm3_seq + sw3
        norm2      = self.norm_2(fm3_skip)
        cls2       = Rearrange('b c 1 -> b 1 c')(self.avg_2(norm2.transpose(1, 2)))

        return [
            torch.cat((cls1, sw1_skip),  dim=1),   # stream 0: fine-grained
            torch.cat((cls2, fm3_skip), dim=1),    # stream 1: semantic
        ]


import os  # needed for os.path.exists above


# ─────────────────────────────────────────────────────────────
# All2Cross encoder
# ─────────────────────────────────────────────────────────────

class All2Cross(nn.Module):
    def __init__(self, config, img_size=224, in_chans=3,
                 embed_dim=(96, 384), norm_layer=nn.LayerNorm):
        super().__init__()
        self.cross_pos_embed = config.cross_pos_embed
        self.pyramid         = PyramidFeatures(config=config, img_size=img_size,
                                               in_channels=in_chans)

        n_p1 = (img_size // config.patch_size) ** 2
        n_p2 = (img_size // config.patch_size // 4) ** 2
        num_patches      = (n_p1, n_p2)
        self.num_branches = 2

        self.pos_embed = nn.ParameterList([
            nn.Parameter(torch.zeros(1, 1 + num_patches[i], embed_dim[i]))
            for i in range(self.num_branches)
        ])

        total_depth = sum([max(x[:-1]) + x[-1] for x in config.depth])
        dpr     = [x.item() for x in torch.linspace(0, config.drop_path_rate, total_depth)]
        dpr_ptr = 0
        self.blocks = nn.ModuleList()
        for block_config in config.depth:
            curr_depth = max(block_config[:-1]) + block_config[-1]
            blk = MultiScaleBlock(
                embed_dim, num_patches, block_config,
                num_heads=config.num_heads[:2],
                mlp_ratio=config.mlp_ratio,
                qkv_bias=config.qkv_bias, qk_scale=config.qk_scale,
                drop=config.drop_rate, attn_drop=config.attn_drop_rate,
                drop_path=dpr[dpr_ptr:dpr_ptr + curr_depth],
                norm_layer=norm_layer,
            )
            dpr_ptr += curr_depth
            self.blocks.append(blk)

        self.norm = nn.ModuleList([norm_layer(embed_dim[i]) for i in range(self.num_branches)])
        for i in range(self.num_branches):
            if self.pos_embed[i].requires_grad:
                trunc_normal_(self.pos_embed[i], std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        xs = self.pyramid(x)
        if self.cross_pos_embed:
            for i in range(self.num_branches):
                xs[i] = xs[i] + self.pos_embed[i]
        for blk in self.blocks:
            xs = blk(xs)
        return [self.norm[i](xs[i]) for i in range(self.num_branches)]


# ─────────────────────────────────────────────────────────────
# Decoder
# ─────────────────────────────────────────────────────────────

class ConvUpsample(nn.Module):
    def __init__(self, in_chans=384, out_chans=(128,), upsample=True):
        super().__init__()
        tower = []
        curr  = in_chans
        for out_ch in out_chans:
            tower += [
                nn.Conv2d(curr, out_ch, 3, padding=1, bias=False),
                nn.GroupNorm(32, out_ch),
                nn.ReLU(inplace=False),
            ]
            if upsample:
                tower.append(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False))
            curr = out_ch
        self.convs_level = nn.Sequential(*tower)

    def forward(self, x):
        return self.convs_level(x)


class SegmentationHead(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size // 2)
        )


# ─────────────────────────────────────────────────────────────
# Full HiFormer model
# ─────────────────────────────────────────────────────────────

class HiFormer(nn.Module):
    """
    HiFormer for medical image segmentation.

    Args:
        config     : HiFormerConfig instance
        img_size   : input image size (square), default 224
        in_chans   : number of input channels, default 3
        n_classes  : number of output classes (1 for binary segmentation)
    """
    def __init__(self, config, img_size=224, in_chans=3, n_classes=1):
        super().__init__()
        self.img_size   = img_size
        self.patch_size = [4, 16]
        self.n_classes  = n_classes

        embed_dim = (config.swin_pyramid_fm[0], config.swin_pyramid_fm[2])

        self.All2Cross = All2Cross(config=config, img_size=img_size,
                                   in_chans=in_chans, embed_dim=embed_dim)
        self.ConvUp_s  = ConvUpsample(in_chans=embed_dim[1],
                                      out_chans=(128, 128), upsample=True)
        self.ConvUp_l  = ConvUpsample(in_chans=embed_dim[0],
                                      out_chans=(128,),     upsample=False)

        self.conv_pred = nn.Sequential(
            nn.Conv2d(128, 16, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False),
        )
        self.segmentation_head = SegmentationHead(16, n_classes, kernel_size=3)

    def forward(self, x):
        xs        = self.All2Cross(x)
        embeddings = [t[:, 1:] for t in xs]   # strip CLS token
        reshaped   = []
        for i, emb in enumerate(embeddings):
            h = w = self.img_size // self.patch_size[i]
            emb = Rearrange('b (h w) d -> b d h w', h=h, w=w)(emb)
            emb = self.ConvUp_l(emb) if i == 0 else self.ConvUp_s(emb)
            reshaped.append(emb)
        C   = reshaped[0] + reshaped[1]
        C   = self.conv_pred(C)
        out = self.segmentation_head(C)
        if self.n_classes == 1:
            out = torch.sigmoid(out)
        return out
