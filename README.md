# Medical Image Segmentation — ISIC Skin Lesion Dataset

Comparative study of three deep learning architectures for skin lesion segmentation:

| Model | Architecture | Backbone |
|---|---|---|
| **U-Net** | Encoder-Decoder with skip connections | From scratch |
| **Attention DeepLabV3+** | ASPP + Branch & Channel Attention | Xception65 (ImageNet) |
| **HiFormer** | CNN + Swin Transformer cross-attention | ResNet50 + Swin-T (ImageNet) |

---

## Project Structure

```
medical-segmentation-isic/
├── prepare_dataset.py                  # Dataset preparation (ISIC 2017 or 2018)
├── evaluate.py                         # Unified evaluation for all models
├── requirements.txt
│
├── data/                               # .npy files go here (created by prepare_dataset.py)
│   ├── data_train.npy
│   ├── data_val.npy
│   ├── data_test.npy
│   ├── mask_train.npy
│   ├── mask_val.npy
│   └── mask_test.npy
│
├── unet/
│   ├── unet_model.py                   # U-Net architecture + loss functions
│   └── train_unet.py                   # Training script
│
├── attention_deeplabv3plus/
│   ├── models.py                       # AttentionDeeplabv3+ architecture
│   └── train.py                        # Training script (2-phase warmup + fine-tune)
│
├── hiformer/
│   ├── hiformer_model.py               # HiFormer architecture
│   └── train_hiformer.py              # Training script
│
└── results/                            # Auto-created during training
    ├── unet/
    ├── deeplab/
    ├── hiformer/
    └── eval/

# ISIC 2017
python prepare_dataset.py \
    --dataset_root /path/to/isic2017 \
    --year 2017 \
    --out_dir data
```

This produces six `.npy` files inside `data/`:

| File | Shape | Description |
|---|---|---|
| `data_train.npy` | (1815, 256, 256, 3) | Training images |
| `data_val.npy` | (259, 256, 256, 3) | Validation images |
| `data_test.npy` | (520, 256, 256, 3) | Test images |
| `mask_train.npy` | (1815, 256, 256) | Training masks |
| `mask_val.npy` | (259, 256, 256) | Validation masks |
| `mask_test.npy` | (520, 256, 256) | Test masks |

---

## Training

### U-Net (Baseline)

```bash
cd unet
python train_unet.py
```

- **Loss**: Dice loss
- **Optimizer**: Adam (LR=1e-4)
- **Scheduler**: ReduceLROnPlateau (factor=0.5, patience=5)
- **Early stopping**: patience=15 epochs
- **Augmentation**: horizontal/vertical flip + random 90° rotation

### Attention DeepLabV3+

```bash
cd attention_deeplabv3plus
python train.py
```

Two-phase training strategy:

| Phase | Epochs | Backbone | LR (heads) |
|---|---|---|---|
| Warmup | 1–5 | **Frozen** | 1e-4 |
| Fine-tune | 6–100 | Unfrozen | backbone: 1e-5 / heads: 1e-4 |

### HiFormer

```bash
# (Optional) Download pretrained Swin-T weights
mkdir pretrained
# Download swin_tiny_patch4_window7_224.pth from:
# https://github.com/microsoft/Swin-Transformer/releases
# and place in pretrained/

cd hiformer
python train_hiformer.py
```

- **Loss**: BCE + Dice
- **Optimizer**: AdamW (LR=1e-4, weight_decay=1e-4)
- **Scheduler**: Cosine annealing
- **Input size**: 224×224 (HiFormer requirement)

---

## Evaluation

Evaluate any trained model on the test set:

```bash
# U-Net
python evaluate.py \
    --model unet \
    --weights unet/best_unet_isic17.pth \
    --data data \
    --out results/eval

# Attention DeepLabV3+
python evaluate.py \
    --model deeplab \
    --weights results/deeplab/weight_isic_deeplab_v3pa.pth \
    --data data \
    --out results/eval

# HiFormer
python evaluate.py \
    --model hiformer \
    --weights results/hiformer/best_hiformer.pth \
    --data data \
    --out results/eval
```

Output per model: `*_test_results.txt`, `*_roc_curve.png`, `*_predictions.png`

---

## Metrics reported

- F1-Score (Dice)
- Sensitivity (Recall)
- Specificity
- Precision
- Accuracy
- Jaccard Index (IoU)
- Housdroff Distance

---


## References

1. Ronneberger et al. — *U-Net: Convolutional Networks for Biomedical Image Segmentation* (2015)
2. Chen et al. — *Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation* (DeepLabV3+, 2018)
3. Heidari et al. — *HiFormer: Hierarchical Multi-scale Representations Using Transformers for Medical Image Segmentation* (2022)
4. ISIC Archive — https://www.isic-archive.com
