# -*- coding: utf-8 -*-
"""
Attention Deeplabv3+ — PyTorch (pretrained backbone)
Uses timm's ImageNet-pretrained Xception65 as the encoder,
exactly as the original paper does.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

def _fixed_padding(x, kernel_size, rate):
    keff      = kernel_size + (kernel_size - 1) * (rate - 1)
    pad_total = keff - 1
    pad_beg   = pad_total // 2
    pad_end   = pad_total - pad_beg
    return F.pad(x, (pad_beg, pad_end, pad_beg, pad_end))


# ─────────────────────────────────────────────────────────────
# Building blocks
# ─────────────────────────────────────────────────────────────

class SepConvBN(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, kernel_size=3,
                 rate=1, depth_activation=False, epsilon=1e-3):
        super().__init__()
        self.stride           = stride
        self.kernel_size      = kernel_size
        self.rate             = rate
        self.depth_activation = depth_activation
        dw_padding = rate * (kernel_size // 2) if stride == 1 else 0
        self.dw    = nn.Conv2d(in_ch, in_ch, kernel_size, stride=stride,
                               padding=dw_padding, dilation=rate,
                               groups=in_ch, bias=False)
        self.bn_dw = nn.BatchNorm2d(in_ch, eps=epsilon)
        self.pw    = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.bn_pw = nn.BatchNorm2d(out_ch, eps=epsilon)

    def forward(self, x):
        if not self.depth_activation:
            x = F.relu(x)
        if self.stride != 1:
            x = _fixed_padding(x, self.kernel_size, self.rate)
        x = self.bn_dw(self.dw(x))
        if self.depth_activation:
            x = F.relu(x)
        x = self.bn_pw(self.pw(x))
        if self.depth_activation:
            x = F.relu(x)
        return x


class BranchAttention(nn.Module):
    def __init__(self, channels=256):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.drop  = nn.Dropout(0.5)
        self.conv3 = nn.Conv2d(channels * 2, channels, 3, padding=1, bias=False)
        self.conv4 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)

    def forward(self, x):
        x1 = F.relu(self.conv1(x))
        x1 = F.relu(self.conv2(x1))
        x1 = self.drop(x1)
        x2 = F.relu(self.conv3(torch.cat([x, x1], dim=1)))
        return F.relu(self.conv4(x2))


class SharedChannelAttention(nn.Module):
    def __init__(self, channels=256, reduction=8):
        super().__init__()
        self.fc0 = nn.Linear(channels, channels, bias=False)
        self.fc1 = nn.Linear(channels, channels // reduction, bias=False)
        self.fc2 = nn.Linear(channels // reduction, channels, bias=False)

    def forward(self, x):
        s = x.mean(dim=[2, 3])
        s = F.relu(self.fc0(s))
        s = F.relu(self.fc1(s))
        s = torch.sigmoid(self.fc2(s))
        return x * s[:, :, None, None]


# ─────────────────────────────────────────────────────────────
# Main model
# ─────────────────────────────────────────────────────────────

class AttentionDeeplabv3p(nn.Module):
    """
    Attention Deeplabv3+ with ImageNet-pretrained Xception65 backbone.

    Training strategy (handled in train.py):
      - Epochs 1-5  : backbone frozen, only ASPP+attention+decoder train at LR=1e-4
      - Epochs 6+   : full network fine-tuned, backbone at LR=1e-5, rest at LR=1e-4
    """
    def __init__(self, in_channels=3, input_hw=(256, 256),
                 num_classes=1, pretrained=True):
        super().__init__()

        # ── Pretrained Xception65 backbone ────────────────────
        self.backbone = timm.create_model(
            'xception65',
            pretrained=pretrained,
            features_only=True,
            out_indices=(1, 4),
        )

        # Probe actual channel sizes dynamically
        with torch.no_grad():
            dummy      = torch.zeros(1, in_channels, *input_hw)
            feats      = self.backbone(dummy)
            skip_ch    = feats[0].shape[1]
            aspp_in_ch = feats[1].shape[1]
        print(f'Backbone → skip_ch: {skip_ch}, aspp_in_ch: {aspp_in_ch}')

        # ── ASPP ──────────────────────────────────────────────
        self.aspp0_conv    = nn.Conv2d(aspp_in_ch, 256, 1, bias=False)
        self.aspp0_bn      = nn.BatchNorm2d(256, eps=1e-5)
        self.aspp1         = SepConvBN(aspp_in_ch, 256, rate=6,
                                       depth_activation=True, epsilon=1e-5)
        self.aspp2         = SepConvBN(aspp_in_ch, 256, rate=12,
                                       depth_activation=True, epsilon=1e-5)
        self.aspp3         = SepConvBN(aspp_in_ch, 256, rate=18,
                                       depth_activation=True, epsilon=1e-5)
        self.img_pool_conv = nn.Conv2d(aspp_in_ch, 256, 1, bias=False)
        self.img_pool_bn   = nn.BatchNorm2d(256, eps=1e-5)

        # ── Attention ─────────────────────────────────────────
        self.branch_attn  = nn.ModuleList([BranchAttention(256) for _ in range(5)])
        self.channel_attn = SharedChannelAttention(256, reduction=8)
        self.conv3d       = nn.Conv3d(256, 256, kernel_size=(5, 1, 1), bias=False)

        self.post_conv1 = nn.Conv2d(256, 256, 3, padding=1, bias=False)
        self.post_conv2 = nn.Conv2d(256, 256, 3, padding=1, bias=False)
        self.post_bn    = nn.BatchNorm2d(256, eps=1e-5)
        self.post_drop  = nn.Dropout(0.1)

        # ── Decoder ───────────────────────────────────────────
        self.skip_proj    = nn.Conv2d(skip_ch, 48, 1, bias=False)
        self.skip_proj_bn = nn.BatchNorm2d(48, eps=1e-5)
        self.dec_conv0    = SepConvBN(256 + 48, 256, depth_activation=True, epsilon=1e-5)
        self.dec_conv1    = SepConvBN(256,      256, depth_activation=True, epsilon=1e-5)
        self.final1       = nn.Conv2d(256, 64, 3, padding=1)
        self.final2       = nn.Conv2d(64,  64, 3, padding=1)
        self.final3       = nn.Conv2d(64,   2, 3, padding=1)
        self.final4       = nn.Conv2d(2, num_classes, 1)

        self._init_new_weights()

    def _init_new_weights(self):
        for name, m in self.named_modules():
            if name.startswith('backbone'):
                continue
            if isinstance(m, (nn.Conv2d, nn.Conv3d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def freeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = False

    def unfreeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = True

    def forward(self, x):
        h, w = x.shape[2], x.shape[3]

        feats = self.backbone(x)
        skip1 = feats[0]    # (B, 128,  H/4,  W/4)
        x     = feats[1]    # (B, 2048, H/32, W/32)

        # ASPP
        b0 = F.relu(self.aspp0_bn(self.aspp0_conv(x)))
        b1 = self.aspp1(x)
        b2 = self.aspp2(x)
        b3 = self.aspp3(x)
        b4 = F.adaptive_avg_pool2d(x, 1)
        b4 = F.relu(self.img_pool_bn(self.img_pool_conv(b4)))
        b4 = F.interpolate(b4, size=(x.shape[2], x.shape[3]),
                           mode='bilinear', align_corners=True)

        # Attention
        attended = []
        for i, b in enumerate([b0, b1, b2, b3, b4]):
            b = self.branch_attn[i](b)
            b = self.channel_attn(b)
            attended.append(b)

        stacked = torch.stack(attended, dim=2)
        out     = self.conv3d(stacked).squeeze(2)

        out = F.relu(self.post_conv1(out))
        out = F.relu(self.post_conv2(out))
        out = F.relu(self.post_bn(out))
        out = self.post_drop(out)

        # Decoder
        out   = F.interpolate(out, size=skip1.shape[2:],
                              mode='bilinear', align_corners=True)
        skip1 = F.relu(self.skip_proj_bn(self.skip_proj(skip1)))
        out   = torch.cat([out, skip1], dim=1)
        out   = self.dec_conv0(out)
        out   = self.dec_conv1(out)

        out = F.relu(self.final1(out))
        out = F.relu(self.final2(out))
        out = F.relu(self.final3(out))
        out = torch.sigmoid(self.final4(out))

        return F.interpolate(out, size=(h, w), mode='bilinear', align_corners=True)


def build_model(input_hw=(256, 256), in_channels=3, num_classes=1, pretrained=True):
    return AttentionDeeplabv3p(
        in_channels=in_channels,
        input_hw=input_hw,
        num_classes=num_classes,
        pretrained=pretrained,
    )
