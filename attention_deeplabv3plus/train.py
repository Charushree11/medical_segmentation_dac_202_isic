# -*- coding: utf-8 -*-
"""
Train Attention Deeplabv3+ with pretrained backbone.

Training strategy that matches the paper:
  Phase 1 — Epochs 1-5  (warmup):
      Backbone is FROZEN. Only ASPP + attention + decoder train.
      LR = 1e-4. This lets the new layers stabilise before touching
      the pretrained weights.

  Phase 2 — Epochs 6-100 (fine-tune):
      Full network trains with DIFFERENTIAL learning rates:
        - Backbone:               LR = 1e-5  (10x lower — fine-tune gently)
        - ASPP + attention + decoder: LR = 1e-4

Loss: BCE + Dice  (better for segmentation than BCE alone)
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import attention_deeplabv3plus.models as M

# ─────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────
DEVICE        = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DATA_DIR      = '../data'   # path to .npy files — adjust if needed
BATCH_SIZE    = 20
NUM_EPOCHS    = 100
LR            = 1e-4
WARMUP_EPOCHS = 5
INPUT_SHAPE   = (256, 256)
RESULTS_DIR   = 'results/deeplab'
SAVE_PATH     = os.path.join(RESULTS_DIR, 'weight_isic_deeplab_v3pa.pth')
os.makedirs(RESULTS_DIR, exist_ok=True)
print(f'Using device: {DEVICE}')


# ─────────────────────────────────────────────────────────────
# Loss
# ─────────────────────────────────────────────────────────────
def dice_loss(pred, target, smooth=1e-6):
    pred   = pred.view(-1)
    target = target.view(-1)
    inter  = (pred * target).sum()
    return 1 - (2. * inter + smooth) / (pred.sum() + target.sum() + smooth)

def combined_loss(pred, target):
    return nn.BCELoss()(pred, target) + dice_loss(pred, target)


# ─────────────────────────────────────────────────────────────
# Normalisation
# ─────────────────────────────────────────────────────────────
def dataset_normalized(imgs):
    imgs_std  = np.std(imgs)
    imgs_mean = np.mean(imgs)
    imgs_norm = (imgs - imgs_mean) / imgs_std
    for i in range(imgs_norm.shape[0]):
        lo, hi = imgs_norm[i].min(), imgs_norm[i].max()
        imgs_norm[i] = (imgs_norm[i] - lo) / (hi - lo + 1e-8) * 255.0
    return imgs_norm


# ─────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────
class SkinLesionDataset(Dataset):
    def __init__(self, images, masks):
        self.images = torch.from_numpy(images).float()
        self.masks  = torch.from_numpy(masks).float()

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, idx):
        img  = self.images[idx].permute(2, 0, 1) / 255.0
        mask = self.masks[idx].permute(2, 0, 1)
        return img, mask


# ─────────────────────────────────────────────────────────────
# Load data
# ─────────────────────────────────────────────────────────────
print('Loading data …')
tr_data  = np.load(os.path.join(DATA_DIR, 'data_train.npy'))
val_data = np.load(os.path.join(DATA_DIR, 'data_val.npy'))
tr_mask  = np.load(os.path.join(DATA_DIR, 'mask_train.npy'))
val_mask = np.load(os.path.join(DATA_DIR, 'mask_val.npy'))

tr_mask  = np.expand_dims(tr_mask,  axis=3)
val_mask = np.expand_dims(val_mask, axis=3)

tr_data  = dataset_normalized(tr_data)
val_data = dataset_normalized(val_data)
tr_mask  = tr_mask  / 255.0
val_mask = val_mask / 255.0
print('Data loaded and normalised.')

train_loader = DataLoader(SkinLesionDataset(tr_data,  tr_mask),
                          batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=4, pin_memory=True)
val_loader   = DataLoader(SkinLesionDataset(val_data, val_mask),
                          batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=4, pin_memory=True)


# ─────────────────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────────────────
model = M.build_model(input_hw=INPUT_SHAPE, pretrained=True).to(DEVICE)
print(f'Total parameters : {sum(p.numel() for p in model.parameters()):,}')
print(f'Trainable params : {sum(p.numel() for p in model.parameters() if p.requires_grad):,}')

model.freeze_backbone()
print(f'\nPhase 1: backbone frozen for {WARMUP_EPOCHS} warmup epochs.')
print(f'Trainable params now: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}')


def make_optimizer(model, phase):
    if phase == 1:
        return Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)
    else:
        backbone_params = list(model.backbone.parameters())
        backbone_ids    = set(id(p) for p in backbone_params)
        new_params      = [p for p in model.parameters() if id(p) not in backbone_ids]
        return Adam([
            {'params': backbone_params, 'lr': LR * 0.1},
            {'params': new_params,      'lr': LR},
        ])


optimizer     = make_optimizer(model, phase=1)
scheduler     = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=7)
best_val_loss = float('inf')
current_phase = 1

# ─────────────────────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────────────────────
for epoch in range(1, NUM_EPOCHS + 1):

    # Phase switch
    if epoch == WARMUP_EPOCHS + 1 and current_phase == 1:
        model.unfreeze_backbone()
        optimizer = make_optimizer(model, phase=2)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=7)
        current_phase = 2
        print(f'\nPhase 2: backbone unfrozen, differential LR active '
              f'(backbone={LR*0.1:.0e}, heads={LR:.0e})')

    # Train
    model.train()
    train_loss = 0.0
    for imgs, masks in train_loader:
        imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
        optimizer.zero_grad()
        loss = combined_loss(model(imgs), masks)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * imgs.size(0)
    train_loss /= len(train_loader.dataset)

    # Validate
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for imgs, masks in val_loader:
            imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
            val_loss += combined_loss(model(imgs), masks).item() * imgs.size(0)
    val_loss /= len(val_loader.dataset)

    scheduler.step(val_loss)

    tag = ''
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), SAVE_PATH)
        tag = '  ← best saved'

    phase_tag = 'warmup' if current_phase == 1 else 'finetune'
    print(f'Epoch {epoch:3d}/{NUM_EPOCHS} [{phase_tag}] | '
          f'train: {train_loss:.4f} | val: {val_loss:.4f}{tag}')

print(f'\nDone. Best val_loss: {best_val_loss:.4f} → {SAVE_PATH}')
