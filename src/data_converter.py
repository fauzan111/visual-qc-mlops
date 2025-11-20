import os
import cv2
import numpy as np
import shutil
import stat
import sys
import time
from pathlib import Path
from sklearn.model_selection import train_test_split
import random

# --- Configuration ---
CATEGORY = 'bottle' 
DATA_ROOT = Path('./data')
RAW_DIR = DATA_ROOT / 'raw_mvtec' / CATEGORY
YOLO_DIR = DATA_ROOT / 'yolo_dataset'

DEFECT_CLASS_ID = 0 
TEST_SIZE = 0.2    
SEED = 42

random.seed(SEED)

# --- Helper Functions ---

def normalize_bbox(x, y, w, h, img_w, img_h):
    x_center = (x + w / 2) / img_w
    y_center = (y + h / 2) / img_h
    w_norm = w / img_w
    h_norm = h / img_h
    return f"{DEFECT_CLASS_ID} {x_center} {y_center} {w_norm} {h_norm}"

def process_mask_to_yolo(image_path, mask_path, label_file):
    img = cv2.imread(str(image_path))
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    
    if img is None or mask is None: 
        return False

    # --- ROBUST MASK FIX ---
    max_val = mask.max()
    if max_val == 0: return True 
    if max_val <= 1: mask = mask * 255
    
    _, mask = cv2.threshold(mask, 10, 255, cv2.THRESH_BINARY)
    kernel = np.ones((3,3), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=1)

    img_h, img_w, _ = img.shape
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    yolo_annotations = []

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w < 2 or h < 2: continue
        yolo_line = normalize_bbox(x, y, w, h, img_w, img_h)
        yolo_annotations.append(yolo_line)

    with open(label_file, 'w') as f:
        f.write('\n'.join(yolo_annotations))
        
    if len(yolo_annotations) > 0 and random.random() < 0.05:
        print(f"DEBUG: Found {len(yolo_annotations)} defect(s) in {image_path.name}")

    return True 

def robust_copy(src, dst):
    src = Path(src)
    dst = Path(dst)
    
    if dst.exists():
        try:
            os.chmod(dst, stat.S_IWRITE)
            os.remove(dst)
        except Exception:
            pass 
            
    max_retries = 3
    for i in range(max_retries):
        try:
            shutil.copy2(src, dst)
            return
        except PermissionError:
            if i < max_retries - 1:
                time.sleep(0.2)
            else:
                print(f"Warning: Failed to copy {src.name}")

def on_rm_error(func, path, exc_info):
    os.chmod(path, stat.S_IWRITE)
    try:
        func(path)
    except Exception:
        pass

def setup_directories():
    print("Setting up YOLO directory structure...")
    subdirs = {
        'train_img': YOLO_DIR / 'images' / 'train',
        'val_img': YOLO_DIR / 'images' / 'val',
        'train_lbl': YOLO_DIR / 'labels' / 'train',
        'val_lbl': YOLO_DIR / 'labels' / 'val',
        'temp_img': YOLO_DIR / 'temp_images',
        'temp_lbl': YOLO_DIR / 'temp_labels'
    }
    if YOLO_DIR.exists():
        try:
            shutil.rmtree(YOLO_DIR, onerror=on_rm_error)
        except Exception:
            pass 
    for d in subdirs.values():
        d.mkdir(parents=True, exist_ok=True)
    return subdirs

def convert_and_split_dataset():
    subdirs = setup_directories()
    all_stems = [] 
    
    # --- Part 1: Process Defect Images ---
    defect_img_folder = RAW_DIR / 'test'
    print(f"\nProcessing defect images for category: {CATEGORY}...")
    
    for defect_type_folder in defect_img_folder.iterdir():
        if not defect_type_folder.is_dir() or defect_type_folder.name == 'good':
            continue
            
        gt_folder = RAW_DIR / 'ground_truth' / defect_type_folder.name
        image_files = list(defect_type_folder.glob('*.png'))
        
        for img_path in image_files:
            # FIX: Add prefix to avoid overwriting (e.g. broken_large_000.png)
            stem = f"{defect_type_folder.name}_{img_path.stem}"
            
            mask_path = gt_folder / f"{img_path.stem}_mask.png"
            
            temp_img_path = subdirs['temp_img'] / f"{stem}.png"
            temp_label_path = subdirs['temp_lbl'] / f"{stem}.txt" 

            if mask_path.exists():
                process_mask_to_yolo(img_path, mask_path, temp_label_path)
                robust_copy(img_path, temp_img_path)
                all_stems.append(stem)
            
    print(f"Successfully processed defective images.")

    # --- Part 2: Process Normal Images ---
    normal_img_folder = RAW_DIR / 'train' / 'good'
    normal_images = list(normal_img_folder.glob('*.png'))
    
    for img_path in normal_images:
        # FIX: Add 'good' prefix to avoid overwriting
        stem = f"good_{img_path.stem}"
        
        temp_img_path = subdirs['temp_img'] / f"{stem}.png"
        temp_label_path = subdirs['temp_lbl'] / f"{stem}.txt" 

        with open(temp_label_path, 'w') as f:
            f.write('')
        
        robust_copy(img_path, temp_img_path)
        all_stems.append(stem)
        
    print(f"Added {len(normal_images)} defect-free images. Total unique samples: {len(all_stems)}.")
    
    # --- Part 3: Split and Move ---
    train_stems, val_stems = train_test_split(all_stems, test_size=TEST_SIZE, random_state=SEED)
    print(f"Split: Train ({len(train_stems)}), Val ({len(val_stems)})")

    def move_files(stems, img_dest, lbl_dest):
        for stem in stems:
            src_img = subdirs['temp_img'] / f"{stem}.png"
            dst_img = img_dest / f"{stem}.png"
            robust_copy(src_img, dst_img)
            
            src_lbl = subdirs['temp_lbl'] / f"{stem}.txt"
            dst_lbl = lbl_dest / f"{stem}.txt"
            robust_copy(src_lbl, dst_lbl)

    move_files(train_stems, subdirs['train_img'], subdirs['train_lbl'])
    move_files(val_stems, subdirs['val_img'], subdirs['val_lbl'])
    
    try:
        shutil.rmtree(subdirs['temp_img'], onerror=on_rm_error)
        shutil.rmtree(subdirs['temp_lbl'], onerror=on_rm_error)
    except Exception:
        pass
    
    print("\nData conversion complete!")

if __name__ == "__main__":
    if not RAW_DIR.exists():
        print(f"FATAL ERROR: Raw data not found at {RAW_DIR}")
    else:
        convert_and_split_dataset()