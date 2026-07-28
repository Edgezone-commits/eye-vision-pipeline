"""
preprocess.py
-------------
Basic preprocessing for retinal fundus images before they're used for
training or prediction.

Steps:
  1. Resize to a consistent square size (models & Custom Vision both
     benefit from consistent input dimensions).
  2. Center-crop to remove excess black background common in fundus
     photos (the circular retina sits in a black square frame).
  3. Mild contrast enhancement (CLAHE) since many fundus images are
     unevenly lit.

Usage:
    python preprocess.py --input_dir data/raw --output_dir data/processed
"""

import argparse
import os
import cv2
import numpy as np


def resize_and_crop(img, size: int = 512):
    h, w = img.shape[:2]
    # Crop to a centered square first (removes uneven black borders)
    min_dim = min(h, w)
    top = (h - min_dim) // 2
    left = (w - min_dim) // 2
    cropped = img[top: top + min_dim, left: left + min_dim]
    return cv2.resize(cropped, (size, size), interpolation=cv2.INTER_AREA)


def enhance_contrast(img):
    # CLAHE works on a single channel, so apply it on the L channel of LAB color space
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l)
    merged = cv2.merge((l_enhanced, a, b))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)


def process_image(path_in: str, path_out: str, size: int = 512):
    img = cv2.imread(path_in)
    if img is None:
        print(f"  [skip] Could not read {path_in}")
        return False
    img = resize_and_crop(img, size)
    img = enhance_contrast(img)
    cv2.imwrite(path_out, img)
    return True


def process_folder(input_dir: str, output_dir: str, size: int = 512):
    os.makedirs(output_dir, exist_ok=True)
    valid_ext = (".jpg", ".jpeg", ".png")
    count = 0
    for fname in os.listdir(input_dir):
        if fname.lower().endswith(valid_ext):
            ok = process_image(
                os.path.join(input_dir, fname),
                os.path.join(output_dir, fname),
                size,
            )
            count += int(ok)
    print(f"Processed {count} images -> {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess retinal fundus images.")
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--size", type=int, default=512)
    args = parser.parse_args()
    process_folder(args.input_dir, args.output_dir, args.size)
