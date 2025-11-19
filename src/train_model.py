import os
import cv2
import numpy as np
import shutil
from pathlib import Path
from sklearn.model_selection import train_test_split

# --- Configuration ---
# Choose a single category from the MVTec AD dataset for the project
# Recommended: 'bottle', 'cable', 'metal_nut', or 'pill'
CATEGORY = 'bottle' 

# Path setup (relative to the project root)
DATA_ROOT = Path('./data')
RAW_DIR = DATA_ROOT / 'raw_mvtec' / CATEGORY
YOLO_DIR = DATA_ROOT / 'yolo_dataset'

# Class ID for the defect (YOLO starts class IDs at 0)
DEFECT_CLASS_ID = 0 
TEST_SIZE = 0.2 # 20% for validation/testing split
SEED = 42

def normalize_bbox(x, y, w, h, img_w, img_h):
    """
    Converts pixel (x_min, y_min, x_max, y_max) to normalized YOLO format.
    YOLO format: <class_id> <x_center> <y_center> <width> <height> (all normalized 0-1)
    """
    x_center = (x + w / 2) / img_w
    y_center = (y + h / 2) / img_h
    w_norm = w / img_w
    h_norm = h / img_h
    return f"{DEFECT_CLASS_ID} {x_center} {y_center} {w_norm} {h_norm}"

def process_mask_to_yolo(image_path, mask_path, label_file):
    """
    Loads an image and its defect mask, extracts bounding boxes, and writes 
    them to a YOLO label file.
    """
    # 1. Load image and mask
    img = cv2.imread(str(image_path))
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if img is None or mask is None:
        print(f"Warning: Could not load image or mask for {image_path.name}")
        return False

    img_h, img_w, _ = img.shape
    
    # 2. Find contours (boundaries of the white defect regions in the mask)
    # The mask should be a binary image (0 or 255)
    # RETR_EXTERNAL gets only the outermost contour
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    yolo_annotations = []

    # 3. Process each detected contour
    for contour in contours:
        # Calculate the bounding box (x, y, width, height)
        x, y, w, h = cv2.boundingRect(contour)
        
        # Convert to normalized YOLO format string
        yolo_line = normalize_bbox(x, y, w, h, img_w, img_h)
        yolo_annotations.append(yolo_line)

    # 4. Write annotations to the .txt file
    with open(label_file, 'w') as f:
        f.write('\n'.join(yolo_annotations))
        
    return len(yolo_annotations) > 0 # Return True if defects were found

def setup_directories():
    """Sets up the final YOLO directory structure."""
    print("Setting up YOLO directory structure...")
    
    # Define subfolders for the final YOLO dataset
    subdirs = {
        'train_img': YOLO_DIR / 'images' / 'train',
        'val_img': YOLO_DIR / 'images' / 'val',
        'train_lbl': YOLO_DIR / 'labels' / 'train',
        'val_lbl': YOLO_DIR / 'labels' / 'val'
    }
    
    # Clean up and recreate the main YOLO folder
    if YOLO_DIR.exists():
        shutil.rmtree(YOLO_DIR)
    
    # Create all necessary subdirectories
    for d in subdirs.values():
        d.mkdir(parents=True, exist_ok=True)
        
    return subdirs

def convert_and_split_dataset():
    """Main function to perform conversion, file copying, and train/val split."""
    
    subdirs = setup_directories()
    
    # --- Part 1: Gather File Paths ---
    # Find all defect images and their corresponding masks
    defect_img_folder = RAW_DIR / 'test'
    
    # Dictionary to hold the path pairs: {image_path: mask_path}
    file_paths = {} 
    
    for defect_type_folder in defect_img_folder.iterdir():
        if not defect_type_folder.is_dir():
            continue
            
        gt_folder = defect_type_folder / 'ground_truth'
        
        # Get all original images for this defect type
        image_files = list(defect_type_folder.glob('*.png'))
        
        for img_path in image_files:
            # Construct the mask path
            # Example: '000.png' -> 'ground_truth/000_mask.png'
            mask_path = gt_folder / f"{img_path.stem}_mask.png"
            
            if mask_path.exists():
                file_paths[img_path] = mask_path

    # --- Part 2: Process and Save Annotations ---
    # Store temporary list of successfully annotated files
    annotated_files = [] 
    
    print(f"\nProcessing {len(file_paths)} defective images for category: {CATEGORY}")
    
    for img_path, mask_path in file_paths.items():
        # Temporary label name in the output structure
        temp_label_path = YOLO_DIR / 'labels' / f"{img_path.stem}.txt" 
        
        # Perform the conversion
        if process_mask_to_yolo(img_path, mask_path, temp_label_path):
            annotated_files.append(img_path.stem)
        else:
            # If no defect was found (e.g., if the ground_truth mask was empty), remove the temporary label file
            os.remove(temp_label_path) 
            
    print(f"Successfully annotated {len(annotated_files)} images with defects.")

    # --- Part 3: Handle Normal (Defect-Free) Images ---
    # We use defect-free images from the 'train' folder as additional 'no defect' samples.
    # The label file for these will be empty.
    normal_img_folder = RAW_DIR / 'train' / 'good'
    normal_images = list(normal_img_folder.glob('*.png'))
    
    for img_path in normal_images:
        temp_label_path = YOLO_DIR / 'labels' / f"{img_path.stem}.txt"
        # Create an empty label file for 'good' images
        with open(temp_label_path, 'w') as f:
            f.write('') # An empty .txt file means 'no object detected'
        
        # Copy the image file to the temporary images folder
        shutil.copy(img_path, YOLO_DIR / 'images' / f"{img_path.stem}.png")
        annotated_files.append(img_path.stem)
        
    print(f"Added {len(normal_images)} defect-free images as 'no-defect' samples.")

    # --- Part 4: Train/Validation Split and Copying ---
    all_stems = annotated_files
    
    # Perform the split on the list of file names (stems)
    train_stems, val_stems = train_test_split(
        all_stems, test_size=TEST_SIZE, random_state=SEED
    )
    
    print(f"\nSplit into: Training ({len(train_stems)}), Validation ({len(val_stems)})")

    # Copy files to final train/val directories
    def move_files(stems, img_dest, lbl_dest):
        for stem in stems:
            # Move Image
            src_img = YOLO_DIR / 'images' / f"{stem}.png"
            dst_img = img_dest / f"{stem}.png"
            shutil.move(src_img, dst_img)
            
            # Move Label
            src_lbl = YOLO_DIR / 'labels' / f"{stem}.txt"
            dst_lbl = lbl_dest / f"{stem}.txt"
            shutil.move(src_lbl, dst_lbl)

    # Note: We need to ensure the temporary images and labels folders exist or create them before this step
    temp_img_folder = YOLO_DIR / 'images'
    if not temp_img_folder.exists(): temp_img_folder.mkdir(parents=True)
    temp_lbl_folder = YOLO_DIR / 'labels'
    if not temp_lbl_folder.exists(): temp_lbl_folder.mkdir(parents=True)
    
    # Before the move_files call, we need to ensure the images were copied from RAW_DIR to YOLO_DIR/images
    # Let's clean up and improve the file copying logic here for clarity.
    
    # We will modify the script to copy all files to temporary folders first.
    # Re-running the file gathering and copying for robustness:
    all_files_to_split = list(set(file_paths.keys()) | set(normal_images))
    
    # 1. Copy all images and create all labels in a flattened structure first
    # This step is implicitly handled by the previous loops, but let's make it explicit for 'defect' images
    for img_path, mask_path in file_paths.items(): # Defect images
        shutil.copy(img_path, YOLO_DIR / 'images' / f"{img_path.stem}.png")
        
    # 2. Split and move
    move_files(train_stems, subdirs['train_img'], subdirs['train_lbl'])
    move_files(val_stems, subdirs['val_img'], subdirs['val_lbl'])
    
    # Clean up the temporary flat structure folders
    if temp_img_folder.exists():
        shutil.rmtree(temp_img_folder)
    if temp_lbl_folder.exists():
        shutil.rmtree(temp_lbl_folder)
    
    print("\nData conversion and split complete. YOLO dataset is ready!")


if __name__ == "__main__":
    # You must manually download the MVTec AD dataset and place the 
    # chosen category (e.g., 'bottle') folder inside './data/raw_mvtec/' 
    # before running this script.
    
    if not RAW_DIR.exists():
        print("----------------------------------------------------------------------")
        print(f"ERROR: Raw data not found at {RAW_DIR}")
        print("Please download the MVTec AD dataset and place the 'bottle' folder (or your chosen category) inside ./data/raw_mvtec/")
        print("----------------------------------------------------------------------")
    else:
        convert_and_split_dataset()