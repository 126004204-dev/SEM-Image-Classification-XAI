
!unzip /content/dataset.zip

import os, random
from pathlib import Path
from PIL import Image
import numpy as np
import albumentations as A

BASE = Path("/content/sample_data/dataset")
FOLDERS = {
    "comp_100x": BASE/"Comp-cropped"/"100x",
    "comp_200x": BASE/"Comp-cropped"/"200x",
    "noncomp_100x": BASE/"Non-Comp-cropped"/"100x",
    "noncomp_200x": BASE/"Non-Comp-cropped"/"200x",
}
OUT_BASE = Path("/content/sample_data/dataset_augmented")
OUT_BASE.mkdir(parents=True, exist_ok=True)

pipeline = A.Compose([
    A.RandomResizedCrop(size=(224,224), scale=(0.7,1.0), ratio=(0.75,1.33), p=1.0),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.OneOf([
        A.ElasticTransform(alpha=120, sigma=120*0.05, alpha_affine=120*0.03, p=0.7),
        A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.7),
        A.OpticalDistortion(distort_limit=0.3, shift_limit=0.3, p=0.7),
    ], p=0.7),
    A.OneOf([
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.5),
        A.GaussianBlur(blur_limit=3, p=0.5),
        A.MotionBlur(blur_limit=3, p=0.5),
    ], p=0.5),
    A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.7),
    A.HueSaturationValue(hue_shift_limit=15, sat_shift_limit=25, val_shift_limit=15, p=0.5),
    A.CLAHE(clip_limit=4.0, p=0.5),
    A.RandomGamma(gamma_limit=(80,120), p=0.5),
    A.Sharpen(alpha=(0.2,0.5), lightness=(0.5,1.0), p=0.5),
])

for name, src in FOLDERS.items():
    out_dir = OUT_BASE/name
    out_dir.mkdir(parents=True, exist_ok=True)
    tifs = list(src.glob("*.tif"))
    if not tifs:
        raise FileNotFoundError(f"No .tif files in {src}")
    for i in range(100):
        img = np.array(Image.open(random.choice(tifs)).convert("RGB"))
        aug = pipeline(image=img)["image"]
        Image.fromarray(aug).save(out_dir/f"aug_{i:03d}.png")
    print(f"{name}: 100 images saved → {out_dir}")

import numpy as np
import cv2
import glob
from math import log10, sqrt
from pathlib import Path

def compute_mae(a, b):
    return float(np.mean(np.abs(a - b)))

def compute_mse(a, b):
    return float(np.mean((a - b) ** 2))

def compute_psnr(a, b, max_val=1.0):
    mse = compute_mse(a, b)
    return float('inf') if mse == 0 else 20 * log10(max_val / sqrt(mse))

orig_base = Path("/content/sample_data/dataset")
aug_base  = Path("/content/sample_data/dataset_augmented")

FOLDERS = {
    "comp_100x":    "Comp-cropped/100x",
    "comp_200x":    "Comp-cropped/200x",
    "noncomp_100x": "Non-Comp-cropped/100x",
    "noncomp_200x": "Non-Comp-cropped/200x",
}

print(f"{'Class':<15} {'MAE':>8} {'MSE':>10} {'PSNR':>10}")
print("-"*45)

for name, rel in FOLDERS.items():
    orig_dir = orig_base / rel
    aug_dir  = aug_base  / name

    orig_paths = sorted(glob.glob(str(orig_dir / "*.tif")))
    aug_paths  = sorted(glob.glob(str(aug_dir  / "*.png")))

    n = min(len(orig_paths), len(aug_paths))
    if n == 0:
        print(f"{name:<15} NO IMAGES!")
        continue

    maes, mses, psnrs = [], [], []
    for i in range(n):
        o = cv2.imread(orig_paths[i], cv2.IMREAD_COLOR).astype(np.float32) / 255.0
        a = cv2.imread(aug_paths[i] , cv2.IMREAD_COLOR).astype(np.float32) / 255.0

        if a.shape != o.shape:
            a = cv2.resize(a, (o.shape[1], o.shape[0]), interpolation=cv2.INTER_LINEAR)

        maes.append(compute_mae(o, a))
        mses.append(compute_mse(o, a))
        psnrs.append(compute_psnr(o, a))

    print(f"{name:<15} "
          f"{np.mean(maes):8.4f} "
          f"{np.mean(mses):10.6f} "
          f"{np.mean(psnrs):10.2f}")