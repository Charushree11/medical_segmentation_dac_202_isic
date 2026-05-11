# -*- coding: utf-8 -*-
"""
prepare_dataset.py — Prepare ISIC 2017 or 2018 dataset → .npy arrays

Expected folder layout BEFORE running this script:

  For ISIC 2017:
    dataset/
    ├── ISIC-2017_Training_Data/               ← .jpg images
    ├── ISIC-2017_Training_Part1_GroundTruth/  ← _segmentation.png masks
    ├── ISIC-2017_Validation_Data/
    ├── ISIC-2017_Validation_Part1_GroundTruth/
    └── ISIC-2017_Test_v2_Data/
        └── ISIC-2017_Test_v2_Part1_GroundTruth/

  For ISIC 2018 (single Training split, auto-divided):
    dataset_isic18/
    ├── ISIC2018_Task1-2_Training_Input/        ← 2594 .jpg images
    └── ISIC2018_Task1_Training_GroundTruth/    ← 2594 _segmentation.png masks

Output .npy files are saved to --out_dir (default: ./data/):
    data_train.npy  (N_train, 256, 256, 3)
    data_val.npy    (N_val,   256, 256, 3)
    data_test.npy   (N_test,  256, 256, 3)
    mask_train.npy  (N_train, 256, 256)
    mask_val.npy    (N_val,   256, 256)
    mask_test.npy   (N_test,  256, 256)

Usage:
    # ISIC 2018 (auto-split)
    python prepare_dataset.py --dataset_root /path/to/dataset_isic18 --year 2018

    # ISIC 2017 (pre-split)
    python prepare_dataset.py --dataset_root /path/to/dataset --year 2017
"""

import os
import glob
import argparse
import numpy as np
import cv2

HEIGHT, WIDTH, CHANNELS = 256, 256, 3

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_root', required=True, help='Root folder of the dataset')
parser.add_argument('--year',         default='2018', choices=['2017', '2018'])
parser.add_argument('--out_dir',      default='data', help='Where to save .npy files')
args = parser.parse_args()

os.makedirs(args.out_dir, exist_ok=True)


def read_split(img_dir, mask_dir, mask_suffix='_segmentation.png'):
    img_paths = sorted(glob.glob(os.path.join(img_dir, '*.jpg')))
    N = len(img_paths)
    print(f'  Found {N} images in {img_dir}')
    assert N > 0, f'No .jpg images found in {img_dir}'

    Data  = np.zeros([N, HEIGHT, WIDTH, CHANNELS], dtype=np.float64)
    Label = np.zeros([N, HEIGHT, WIDTH],            dtype=np.float64)

    for idx, img_path in enumerate(img_paths):
        if (idx + 1) % 100 == 0:
            print(f'    {idx+1}/{N}')

        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f'Cannot read: {img_path}')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (WIDTH, HEIGHT), interpolation=cv2.INTER_LINEAR)
        Data[idx] = img.astype(np.float64)

        base      = os.path.splitext(os.path.basename(img_path))[0]
        mask_path = os.path.join(mask_dir, base + mask_suffix)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f'Cannot read mask: {mask_path}')
        mask = cv2.resize(mask, (WIDTH, HEIGHT), interpolation=cv2.INTER_LINEAR)
        Label[idx] = mask.astype(np.float64)

    return Data, Label


if args.year == '2018':
    print('Preparing ISIC 2018 …')
    img_dir  = os.path.join(args.dataset_root, 'ISIC2018_Task1-2_Training_Input')
    mask_dir = os.path.join(args.dataset_root, 'ISIC2018_Task1_Training_GroundTruth')
    Data, Label = read_split(img_dir, mask_dir)

    TRAIN_END = 1815
    VAL_END   = 1815 + 259
    splits = {
        'train': (Data[:TRAIN_END],    Label[:TRAIN_END]),
        'val':   (Data[TRAIN_END:VAL_END], Label[TRAIN_END:VAL_END]),
        'test':  (Data[VAL_END:],      Label[VAL_END:]),
    }

else:  # 2017
    print('Preparing ISIC 2017 …')
    r = args.dataset_root

    train_imgs, train_masks = read_split(
        os.path.join(r, 'ISIC-2017_Training_Data'),
        os.path.join(r, 'ISIC-2017_Training_Part1_GroundTruth'))

    val_imgs, val_masks = read_split(
        os.path.join(r, 'ISIC-2017_Validation_Data'),
        os.path.join(r, 'ISIC-2017_Validation_Part1_GroundTruth'))

    test_imgs, test_masks = read_split(
        os.path.join(r, 'ISIC-2017_Test_v2_Data'),
        os.path.join(r, 'ISIC-2017_Test_v2_Part1_GroundTruth'))

    splits = {
        'train': (train_imgs, train_masks),
        'val':   (val_imgs,   val_masks),
        'test':  (test_imgs,  test_masks),
    }

print('\nSaving .npy files …')
for split, (imgs, masks) in splits.items():
    np.save(os.path.join(args.out_dir, f'data_{split}'),  imgs)
    np.save(os.path.join(args.out_dir, f'mask_{split}'), masks)
    print(f'  data_{split}.npy  {imgs.shape}')
    print(f'  mask_{split}.npy  {masks.shape}')

print(f'\n✅ All files saved to: {os.path.abspath(args.out_dir)}')
