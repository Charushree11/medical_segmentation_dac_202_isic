# train_unet.py  — PyTorch | U-Net on ISIC 2017
import os
import sys
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (f1_score, jaccard_score,
                              confusion_matrix, roc_auc_score, roc_curve)
import matplotlib.pyplot as plt

# Allow imports from parent directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from unet.unet_model import UNet, dice_loss, dice_coef, iou_coef

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
BATCH_SIZE    = 8
EPOCHS        = 100
LR            = 1e-4
THRESHOLD     = 0.5
PATIENCE      = 15          # early stopping patience
WEIGHTS_PATH  = 'best_unet_isic17.pth'
RESULTS_DIR   = 'results/unet'
DATA_DIR      = '../data'   # path to .npy files — adjust if needed
os.makedirs(RESULTS_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")


# ─────────────────────────────────────────────
# DATASET
# ─────────────────────────────────────────────
class ISICDataset(Dataset):
    def __init__(self, X, y, augment=False):
        self.X       = X.astype(np.float32) / 255.0
        self.y       = (y > 127).astype(np.float32)
        self.augment = augment

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = torch.tensor(self.X[idx]).permute(2, 0, 1)   # HWC → CHW
        y = torch.tensor(self.y[idx]).unsqueeze(0)        # H,W → 1,H,W

        if self.augment:
            if torch.rand(1) > 0.5:
                x = torch.flip(x, dims=[2])
                y = torch.flip(y, dims=[2])
            if torch.rand(1) > 0.5:
                x = torch.flip(x, dims=[1])
                y = torch.flip(y, dims=[1])
            k = torch.randint(0, 4, (1,)).item()
            x = torch.rot90(x, k, dims=[1, 2])
            y = torch.rot90(y, k, dims=[1, 2])

        return x, y


# ─────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────
print("\nLoading .npy files...")
X_train = np.load(os.path.join(DATA_DIR, 'data_train.npy'))
X_val   = np.load(os.path.join(DATA_DIR, 'data_val.npy'))
X_test  = np.load(os.path.join(DATA_DIR, 'data_test.npy'))
y_train = np.load(os.path.join(DATA_DIR, 'mask_train.npy'))
y_val   = np.load(os.path.join(DATA_DIR, 'mask_val.npy'))
y_test  = np.load(os.path.join(DATA_DIR, 'mask_test.npy'))

print(f"  Train : {X_train.shape} | Val: {X_val.shape} | Test: {X_test.shape}")

train_loader = DataLoader(ISICDataset(X_train, y_train, augment=True),
                          batch_size=BATCH_SIZE, shuffle=True,  num_workers=2, pin_memory=True)
val_loader   = DataLoader(ISICDataset(X_val, y_val,   augment=False),
                          batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)


# ─────────────────────────────────────────────
# MODEL, OPTIMIZER, SCHEDULER
# ─────────────────────────────────────────────
model     = UNet().to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=LR)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-7
)

total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\nModel parameters: {total_params:,}")


# ─────────────────────────────────────────────
# TRAINING LOOP
# ─────────────────────────────────────────────
history = {'train_loss': [], 'val_loss': [], 'val_dice': [], 'val_iou': []}
best_val_dice     = 0.0
epochs_no_improve = 0

print("\n" + "="*60)
print("Starting Training...")
print("="*60)

for epoch in range(1, EPOCHS + 1):

    # ── Train ──────────────────────────────────
    model.train()
    train_loss = 0.0
    for x, y in train_loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        pred = model(x)
        loss = dice_loss(pred, y)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    train_loss /= len(train_loader)

    # ── Validate ───────────────────────────────
    model.eval()
    val_loss = val_dice = val_iou = 0.0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            pred     = model(x)
            val_loss += dice_loss(pred, y).item()
            val_dice += dice_coef(pred, y).item()
            val_iou  += iou_coef(pred, y).item()
    val_loss /= len(val_loader)
    val_dice /= len(val_loader)
    val_iou  /= len(val_loader)

    scheduler.step(val_loss)

    history['train_loss'].append(train_loss)
    history['val_loss'].append(val_loss)
    history['val_dice'].append(val_dice)
    history['val_iou'].append(val_iou)

    current_lr = optimizer.param_groups[0]['lr']
    print(f"Epoch [{epoch:3d}/{EPOCHS}]  "
          f"Train Loss: {train_loss:.4f}  |  "
          f"Val Loss: {val_loss:.4f}  |  "
          f"Val Dice: {val_dice:.4f}  |  "
          f"Val IoU: {val_iou:.4f}  |  "
          f"LR: {current_lr:.2e}")

    if val_dice > best_val_dice:
        best_val_dice = val_dice
        torch.save(model.state_dict(), WEIGHTS_PATH)
        print(f"  ✅ Best model saved (Val Dice: {best_val_dice:.4f})")
        epochs_no_improve = 0
    else:
        epochs_no_improve += 1

    if epochs_no_improve >= PATIENCE:
        print(f"\n⏹  Early stopping at epoch {epoch}")
        break


# ─────────────────────────────────────────────
# EVALUATION ON TEST SET
# ─────────────────────────────────────────────
print("\n" + "="*60)
print("Evaluating on Test Set (best weights)...")
print("="*60)

model.load_state_dict(torch.load(WEIGHTS_PATH))
model.eval()

test_dataset = ISICDataset(X_test, y_test, augment=False)
test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

all_preds, all_probs, all_trues = [], [], []

with torch.no_grad():
    for x, y in test_loader:
        x    = x.to(DEVICE)
        prob = model(x).cpu().numpy()
        all_probs.append(prob)
        all_preds.append((prob > THRESHOLD).astype(np.uint8))
        all_trues.append(y.numpy().astype(np.uint8))

y_prob = np.concatenate(all_probs, axis=0)
y_pred = np.concatenate(all_preds, axis=0)
y_true = np.concatenate(all_trues, axis=0)

yp_flat   = y_pred.flatten()
yt_flat   = y_true.flatten()
prob_flat = y_prob.flatten()

tn, fp, fn, tp = confusion_matrix(yt_flat, yp_flat).ravel()
f1          = f1_score(yt_flat, yp_flat)
iou         = jaccard_score(yt_flat, yp_flat)
sensitivity = tp / (tp + fn + 1e-8)
specificity = tn / (tn + fp + 1e-8)
precision   = tp / (tp + fp + 1e-8)
accuracy    = (tp + tn) / (tp + tn + fp + fn + 1e-8)
auc         = roc_auc_score(yt_flat, prob_flat)

print(f"\n{'='*55}")
print(f"       U-Net Results on ISIC 2017 Test Set")
print(f"{'='*55}")
print(f"  F1-Score    (Dice)  : {f1:.4f}   | Paper: 0.8682")
print(f"  Sensitivity (Recall): {sensitivity:.4f}   | Paper: 0.9479")
print(f"  Specificity         : {specificity:.4f}   | Paper: 0.9263")
print(f"  Precision           : {precision:.4f}")
print(f"  Accuracy            : {accuracy:.4f}   | Paper: 0.9314")
print(f"  Jaccard (IoU)       : {iou:.4f}   | Paper: 0.9314")
print(f"  AUC                 : {auc:.4f}")
print(f"{'='*55}")

with open(os.path.join(RESULTS_DIR, 'test_results.txt'), 'w') as f:
    f.write("U-Net Results on ISIC 2017\n" + "="*40 + "\n")
    f.write(f"F1-Score    : {f1:.4f}\n")
    f.write(f"Sensitivity : {sensitivity:.4f}\n")
    f.write(f"Specificity : {specificity:.4f}\n")
    f.write(f"Precision   : {precision:.4f}\n")
    f.write(f"Accuracy    : {accuracy:.4f}\n")
    f.write(f"Jaccard     : {iou:.4f}\n")
    f.write(f"AUC         : {auc:.4f}\n")

# ── Plots ──────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
axes[0].plot(history['train_loss'], label='Train Loss')
axes[0].plot(history['val_loss'],   label='Val Loss')
axes[0].set_title('Dice Loss'); axes[0].set_xlabel('Epoch')
axes[0].legend(); axes[0].grid(True)
axes[1].plot(history['val_dice'], label='Val Dice', color='green')
axes[1].set_title('Validation Dice Coefficient'); axes[1].set_xlabel('Epoch')
axes[1].legend(); axes[1].grid(True)
axes[2].plot(history['val_iou'], label='Val IoU', color='orange')
axes[2].set_title('Validation IoU'); axes[2].set_xlabel('Epoch')
axes[2].legend(); axes[2].grid(True)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'training_curves.png'), dpi=150)
plt.close()

fpr, tpr, _ = roc_curve(yt_flat, prob_flat)
plt.figure(figsize=(7, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'U-Net (AUC = {auc:.4f})')
plt.plot([0,1],[0,1], 'navy', lw=1.5, linestyle='--', label='Random')
plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
plt.title('ROC Curve — U-Net on ISIC 2017')
plt.legend(loc='lower right'); plt.grid(True)
plt.savefig(os.path.join(RESULTS_DIR, 'roc_curve.png'), dpi=150)
plt.close()

num_samples = 8
indices = np.random.choice(len(X_test), num_samples, replace=False)
X_test_norm = X_test.astype(np.float32) / 255.0
fig, axes = plt.subplots(num_samples, 3, figsize=(10, num_samples * 3))
fig.suptitle('U-Net Predictions — ISIC 2017', fontsize=14)
for col, title in enumerate(['Input Image', 'Ground Truth', 'Predicted Mask']):
    axes[0, col].set_title(title, fontsize=12)
for row, idx in enumerate(indices):
    axes[row, 0].imshow(X_test_norm[idx])
    axes[row, 1].imshow(y_true[idx, 0], cmap='gray')
    axes[row, 2].imshow(y_pred[idx, 0], cmap='gray')
    for col in range(3):
        axes[row, col].axis('off')
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'predictions.png'), dpi=150, bbox_inches='tight')
plt.close()

print(f"\n✅ All results saved to '{RESULTS_DIR}/'")
