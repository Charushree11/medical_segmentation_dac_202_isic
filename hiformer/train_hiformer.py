# -*- coding: utf-8 -*-
"""
Train HiFormer for skin lesion segmentation on ISIC dataset.

Loss  : BCE + Dice
Optim : AdamW with cosine-annealing LR schedule
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hiformer.hiformer_model import HiFormer, get_hiformer_config

# ─────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────
DEVICE      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DATA_DIR    = '../data'
RESULTS_DIR = 'results/hiformer'
SAVE_PATH   = os.path.join(RESULTS_DIR, 'best_hiformer.pth')
os.makedirs(RESULTS_DIR, exist_ok=True)

IMG_SIZE    = 224
BATCH_SIZE  = 8
NUM_EPOCHS  = 100
LR          = 1e-4
PATIENCE    = 20
N_CLASSES   = 1

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
# Dataset
# ─────────────────────────────────────────────────────────────
def dataset_normalized(imgs):
    imgs_std  = np.std(imgs)
    imgs_mean = np.mean(imgs)
    imgs_norm = (imgs - imgs_mean) / imgs_std
    for i in range(imgs_norm.shape[0]):
        lo, hi = imgs_norm[i].min(), imgs_norm[i].max()
        imgs_norm[i] = (imgs_norm[i] - lo) / (hi - lo + 1e-8) * 255.0
    return imgs_norm


class SkinLesionDataset(Dataset):
    def __init__(self, images, masks, img_size=224):
        import torch.nn.functional as F
        # Resize to IMG_SIZE if needed
        self.images = torch.from_numpy(images).float().permute(0, 3, 1, 2) / 255.0  # N,C,H,W
        self.masks  = torch.from_numpy(masks).float().unsqueeze(1)                   # N,1,H,W
        if self.images.shape[-1] != img_size:
            self.images = F.interpolate(self.images, size=(img_size, img_size), mode='bilinear', align_corners=False)
            self.masks  = F.interpolate(self.masks,  size=(img_size, img_size), mode='nearest')

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, idx):
        return self.images[idx], self.masks[idx]


# ─────────────────────────────────────────────────────────────
# Load data
# ─────────────────────────────────────────────────────────────
print('Loading data …')
tr_data  = dataset_normalized(np.load(os.path.join(DATA_DIR, 'data_train.npy')))
val_data = dataset_normalized(np.load(os.path.join(DATA_DIR, 'data_val.npy')))
tr_mask  = np.load(os.path.join(DATA_DIR, 'mask_train.npy')) / 255.0
val_mask = np.load(os.path.join(DATA_DIR, 'mask_val.npy'))   / 255.0

train_loader = DataLoader(SkinLesionDataset(tr_data,  tr_mask,  IMG_SIZE),
                          batch_size=BATCH_SIZE, shuffle=True,  num_workers=4, pin_memory=True)
val_loader   = DataLoader(SkinLesionDataset(val_data, val_mask, IMG_SIZE),
                          batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
print(f'Train: {len(train_loader.dataset)} | Val: {len(val_loader.dataset)}')


# ─────────────────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────────────────
config = get_hiformer_config()
model  = HiFormer(config=config, img_size=IMG_SIZE, in_chans=3,
                  n_classes=N_CLASSES).to(DEVICE)

total = sum(p.numel() for p in model.parameters())
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'Total params: {total:,} | Trainable: {trainable:,}')

optimizer = AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-6)


# ─────────────────────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────────────────────
best_val_loss     = float('inf')
epochs_no_improve = 0

print("\n" + "="*60)
print("Starting HiFormer Training...")
print("="*60)

for epoch in range(1, NUM_EPOCHS + 1):
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

    scheduler.step()
    current_lr = optimizer.param_groups[0]['lr']

    tag = ''
    if val_loss < best_val_loss:
        best_val_loss     = val_loss
        epochs_no_improve = 0
        torch.save(model.state_dict(), SAVE_PATH)
        tag = '  ← best saved'
    else:
        epochs_no_improve += 1

    print(f'Epoch {epoch:3d}/{NUM_EPOCHS} | '
          f'train: {train_loss:.4f} | val: {val_loss:.4f} | '
          f'LR: {current_lr:.2e}{tag}')

    if epochs_no_improve >= PATIENCE:
        print(f'\n⏹  Early stopping at epoch {epoch}')
        break

print(f'\nDone. Best val_loss: {best_val_loss:.4f} → {SAVE_PATH}')
