# unet_model.py
import torch
import torch.nn as nn

# ─────────────────────────────────────────────
# Loss & Metrics
# ─────────────────────────────────────────────
def dice_coef(y_pred, y_true, smooth=1e-6):
    y_pred = y_pred.view(-1)
    y_true = y_true.view(-1)
    intersection = (y_pred * y_true).sum()
    return (2. * intersection + smooth) / (y_pred.sum() + y_true.sum() + smooth)

def dice_loss(y_pred, y_true):
    return 1 - dice_coef(y_pred, y_true)

def iou_coef(y_pred, y_true, smooth=1e-6):
    y_pred = y_pred.view(-1)
    y_true = y_true.view(-1)
    intersection = (y_pred * y_true).sum()
    union = y_pred.sum() + y_true.sum() - intersection
    return (intersection + smooth) / (union + smooth)


# ─────────────────────────────────────────────
# Building Block
# ─────────────────────────────────────────────
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, dropout=0.2):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


# ─────────────────────────────────────────────
# U-Net Architecture
# ─────────────────────────────────────────────
class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, base_filters=64, dropout=0.2):
        super().__init__()
        f = base_filters

        self.pool = nn.MaxPool2d(2)

        # ── Encoder ──────────────────────────
        self.c1 = ConvBlock(in_channels, f,    dropout)   # 256→256, 3→64
        self.c2 = ConvBlock(f,           f*2,  dropout)   # 128→128, 64→128
        self.c3 = ConvBlock(f*2,         f*4,  dropout)   # 64→64,   128→256
        self.c4 = ConvBlock(f*4,         f*8,  dropout)   # 32→32,   256→512
        self.c5 = ConvBlock(f*8,         f*16, dropout)   # 16→16,   512→1024  (bottleneck)

        # ── Decoder ──────────────────────────
        self.u6 = nn.ConvTranspose2d(f*16, f*8, 2, stride=2)
        self.c6 = ConvBlock(f*16, f*8,  dropout)

        self.u7 = nn.ConvTranspose2d(f*8,  f*4, 2, stride=2)
        self.c7 = ConvBlock(f*8,  f*4,  dropout)

        self.u8 = nn.ConvTranspose2d(f*4,  f*2, 2, stride=2)
        self.c8 = ConvBlock(f*4,  f*2,  dropout)

        self.u9 = nn.ConvTranspose2d(f*2,  f,   2, stride=2)
        self.c9 = ConvBlock(f*2,  f,    dropout)

        # ── Output ───────────────────────────
        self.out_conv = nn.Conv2d(f, out_channels, 1)

    def forward(self, x):
        # Encoder
        c1 = self.c1(x)
        c2 = self.c2(self.pool(c1))
        c3 = self.c3(self.pool(c2))
        c4 = self.c4(self.pool(c3))
        c5 = self.c5(self.pool(c4))

        # Decoder + skip connections
        x = self.c6(torch.cat([self.u6(c5), c4], dim=1))
        x = self.c7(torch.cat([self.u7(x),  c3], dim=1))
        x = self.c8(torch.cat([self.u8(x),  c2], dim=1))
        x = self.c9(torch.cat([self.u9(x),  c1], dim=1))

        return torch.sigmoid(self.out_conv(x))
