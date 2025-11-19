"""Convert MVTec-style masks into YOLO-format label files.

The script expects directories of images and corresponding masks with same filenames
or a consistent naming pattern (e.g. mask files named with `_mask` suffix). It
walks `masks_dir` for mask PNGs, finds corresponding images under `images_dir`,
extracts contours from each mask, computes tight bounding boxes, normalizes them
to YOLO format, and writes label files plus copies images into a YOLO-style
folder layout:

    out_dir/
      images/train
      images/val
      labels/train
      labels/val

Each label file contains one line per detected contour: `0 x_center y_center w h`.
"""
from pathlib import Path
import argparse
import shutil
import random
import cv2
import numpy as np
from .utils import ensure_dir


def _find_image_for_mask(mask_path: Path, images_dir: Path, exts=('.png', '.jpg', '.jpeg', '.bmp', '.tif')):
    """Find an image corresponding to a mask file.

    Strategy:
    - If mask filename ends with `_mask` remove suffix and try stem + ext
    - Try common extensions directly under `images_dir`
    - Recursively search `images_dir` for a file with the same stem
    """
    stem = mask_path.stem
    if stem.endswith('_mask'):
        stem = stem[:-5]

    for ext in exts:
        candidate = images_dir / (stem + ext)
        if candidate.exists():
            return candidate

    for p in images_dir.rglob('*'):
        if p.is_file() and p.stem == stem:
            return p

    return None


def _mask_to_bboxes(mask: np.ndarray, min_area: int = 4):
    """Return list of bboxes (x_min, y_min, x_max, y_max) for contours in mask."""
    if mask is None:
        return []
    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)
    _, thresh = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bboxes = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w * h < min_area:
            continue
        bboxes.append((x, y, x + w, y + h))
    return bboxes


def convert(masks_dir: Path, images_dir: Path, out_dir: Path, split_ratio: float = 0.8, seed: int = 42):
    masks_dir = Path(masks_dir)
    images_dir = Path(images_dir)
    out_dir = Path(out_dir)

    images_out_train = out_dir / 'images' / 'train'
    images_out_val = out_dir / 'images' / 'val'
    labels_out_train = out_dir / 'labels' / 'train'
    labels_out_val = out_dir / 'labels' / 'val'

    for p in [images_out_train, images_out_val, labels_out_train, labels_out_val]:
        ensure_dir(p)

    mask_files = sorted([p for p in masks_dir.rglob('*.png')])
    if not mask_files:
        print(f'No mask files found in {masks_dir}')
        return

    pairs = []
    for mask in mask_files:
        img = _find_image_for_mask(mask, images_dir)
        if img is None:
            print(f'Warning: no image found for mask {mask}; skipping')
            continue
        pairs.append((mask, img))

    random.Random(seed).shuffle(pairs)
    split_idx = int(len(pairs) * split_ratio)

    for i, (mask_path, img_path) in enumerate(pairs):
        img = cv2.imread(str(img_path))
        if img is None:
            print(f'Failed to read image {img_path}; skipping')
            continue
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            print(f'Failed to read mask {mask_path}; skipping')
            continue

        h, w = img.shape[:2]
        bboxes = _mask_to_bboxes(mask)
        if not bboxes:
            # no detected defects in mask -> skip
            continue

        lines = []
        for (x_min, y_min, x_max, y_max) in bboxes:
            x_center = (x_min + x_max) / 2.0 / w
            y_center = (y_min + y_max) / 2.0 / h
            width = (x_max - x_min) / float(w)
            height = (y_max - y_min) / float(h)
            lines.append(f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

        if i < split_idx:
            target_img = images_out_train / img_path.name
            target_lbl = labels_out_train / (img_path.stem + '.txt')
        else:
            target_img = images_out_val / img_path.name
            target_lbl = labels_out_val / (img_path.stem + '.txt')

        shutil.copy2(img_path, target_img)
        with open(target_lbl, 'w') as f:
            f.write('\n'.join(lines) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--masks_dir', required=True, help='Root directory of mask files (may contain subfolders)')
    parser.add_argument('--images_dir', required=True, help='Root directory of original images')
    parser.add_argument('--out_dir', required=True, help='Output YOLO dataset directory')
    parser.add_argument('--split', type=float, default=0.8, help='Train split ratio (default 0.8)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for splitting')
    args = parser.parse_args()
    convert(Path(args.masks_dir), Path(args.images_dir), Path(args.out_dir), args.split, args.seed)
