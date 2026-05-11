# -*- coding: utf-8 -*-
"""
evaluate.py — Unified evaluation script for all models.

Usage:
    python evaluate.py --model unet      --weights results/unet/best_unet_isic17.pth
    python evaluate.py --model deeplab   --weights results/deeplab/weight_isic_deeplab_v3pa.pth
    python evaluate.py --model hiformer  --weights results/hiformer/best_hiformer.pth
"""

import os
import sys
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import (f1_score, jaccard_score, confusion_matrix,
                              roc_auc_score, roc_curve)
from torch.utils.data import Dataset, DataLoader

# ─────────────────────────────────────────────────────────────
# Args
# ─────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--model',   required=True, choices=['unet', 'deeplab', 'hiformer'])
parser.add_argument('--weights', required=True, help='Path to saved .pth file')
parser.add_argument('--data',    default='../data',    help='Folder containing .npy files')
parser.add_argument('--out',     default='results/eval', help='Output folder for results')
parser.add_argument('--batch',   type=int, default=8)
parser.add_argument('--threshold', type=float, default=0.5)
args = parser.parse_args()

os.makedirs(args.out, exist_ok=True)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Evaluating [{args.model}] on {DEVICE}')


# ─────────────────────────────────────────────────────────────
# Load model
# ─────────────────────────────────────────────────────────────
if args.model == 'unet':
    from unet.unet_model import UNet
    model = UNet()
    IMG_SIZE = None   # keep original 256

elif args.model == 'deeplab':
    import attention_deeplabv3plus.models as M
    model = M.build_model(input_hw=(256, 256), pretrained=False)
    IMG_SIZE = None

elif args.model == 'hiformer':
    from hiformer.hiformer_model import HiFormer, get_hiformer_config
    config = get_hiformer_config()
    model  = HiFormer(config=config, img_size=224, in_chans=3, n_classes=1)
    IMG_SIZE = 224

model.load_state_dict(torch.load(args.weights, map_location=DEVICE))
model = model.to(DEVICE)
model.eval()
print(f'Weights loaded from {args.weights}')


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


class EvalDataset(Dataset):
    def __init__(self, images, masks, img_size=None):
        import torch.nn.functional as F
        self.images = torch.from_numpy(images).float().permute(0, 3, 1, 2) / 255.0
        self.masks  = torch.from_numpy(masks).float().unsqueeze(1)
        if img_size and self.images.shape[-1] != img_size:
            self.images = F.interpolate(self.images, size=(img_size, img_size),
                                        mode='bilinear', align_corners=False)
            self.masks  = F.interpolate(self.masks, size=(img_size, img_size), mode='nearest')

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, idx):
        return self.images[idx], self.masks[idx]


print('Loading test data …')
X_test = dataset_normalized(np.load(os.path.join(args.data, 'data_test.npy')))
y_test = np.load(os.path.join(args.data, 'mask_test.npy')) / 255.0

test_loader = DataLoader(EvalDataset(X_test, y_test, IMG_SIZE),
                         batch_size=args.batch, shuffle=False)


# ─────────────────────────────────────────────────────────────
# Inference
# ─────────────────────────────────────────────────────────────
all_probs, all_preds, all_trues = [], [], []

with torch.no_grad():
    for imgs, masks in test_loader:
        imgs = imgs.to(DEVICE)
        prob = model(imgs).cpu().numpy()
        all_probs.append(prob)
        all_preds.append((prob > args.threshold).astype(np.uint8))
        all_trues.append(masks.numpy().astype(np.uint8))

y_prob = np.concatenate(all_probs, axis=0)
y_pred = np.concatenate(all_preds, axis=0)
y_true = np.concatenate(all_trues, axis=0)

yp_flat   = y_pred.flatten()
yt_flat   = y_true.flatten()
prob_flat = y_prob.flatten()


# ─────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────
tn, fp, fn, tp = confusion_matrix(yt_flat, yp_flat).ravel()
f1          = f1_score(yt_flat, yp_flat)
iou         = jaccard_score(yt_flat, yp_flat)
sensitivity = tp / (tp + fn + 1e-8)
specificity = tn / (tn + fp + 1e-8)
precision   = tp / (tp + fp + 1e-8)
accuracy    = (tp + tn) / (tp + tn + fp + fn + 1e-8)
auc         = roc_auc_score(yt_flat, prob_flat)

print(f"\n{'='*55}")
print(f"  Results — {args.model.upper()} on ISIC Test Set")
print(f"{'='*55}")
print(f"  F1-Score    (Dice)  : {f1:.4f}")
print(f"  Sensitivity (Recall): {sensitivity:.4f}")
print(f"  Specificity         : {specificity:.4f}")
print(f"  Precision           : {precision:.4f}")
print(f"  Accuracy            : {accuracy:.4f}")
print(f"  Jaccard (IoU)       : {iou:.4f}")
print(f"  AUC                 : {auc:.4f}")
print(f"{'='*55}")

results_file = os.path.join(args.out, f'{args.model}_test_results.txt')
with open(results_file, 'w') as f:
    f.write(f"Results — {args.model.upper()}\n{'='*40}\n")
    f.write(f"F1-Score    : {f1:.4f}\n")
    f.write(f"Sensitivity : {sensitivity:.4f}\n")
    f.write(f"Specificity : {specificity:.4f}\n")
    f.write(f"Precision   : {precision:.4f}\n")
    f.write(f"Accuracy    : {accuracy:.4f}\n")
    f.write(f"Jaccard     : {iou:.4f}\n")
    f.write(f"AUC         : {auc:.4f}\n")
print(f'\nSaved metrics → {results_file}')

# ── ROC Curve ────────────────────────────────────────────────
fpr, tpr, _ = roc_curve(yt_flat, prob_flat)
plt.figure(figsize=(7, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'{args.model.upper()} (AUC={auc:.4f})')
plt.plot([0, 1], [0, 1], 'navy', lw=1.5, linestyle='--', label='Random')
plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
plt.title(f'ROC Curve — {args.model.upper()}')
plt.legend(loc='lower right'); plt.grid(True)
roc_path = os.path.join(args.out, f'{args.model}_roc_curve.png')
plt.savefig(roc_path, dpi=150); plt.close()
print(f'Saved ROC    → {roc_path}')

# ── Visual Predictions ───────────────────────────────────────
num_samples = 8
indices = np.random.choice(len(X_test), num_samples, replace=False)
X_vis   = X_test.astype(np.float32) / 255.0

fig, axes = plt.subplots(num_samples, 3, figsize=(10, num_samples * 3))
fig.suptitle(f'{args.model.upper()} Predictions', fontsize=14)
for col, title in enumerate(['Input Image', 'Ground Truth', 'Predicted Mask']):
    axes[0, col].set_title(title, fontsize=12)
for row, idx in enumerate(indices):
    axes[row, 0].imshow(X_vis[idx])
    axes[row, 1].imshow(y_true[idx, 0], cmap='gray')
    axes[row, 2].imshow(y_pred[idx, 0], cmap='gray')
    for col in range(3):
        axes[row, col].axis('off')
plt.tight_layout()
pred_path = os.path.join(args.out, f'{args.model}_predictions.png')
plt.savefig(pred_path, dpi=150, bbox_inches='tight'); plt.close()
print(f'Saved preds  → {pred_path}')
