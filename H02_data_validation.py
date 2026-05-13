import os
import cv2
import pickle
from imutils import paths
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from collections import Counter

# Read the training folder path written by 01_extract_and_balance.py
with open('train_folder.txt', 'r') as f:
    target_dir = f.read().strip()

IMG_SIZE = 224

print(f"Loading images from: {target_dir}")
image_paths = list(paths.list_images(target_dir))
labels = [os.path.basename(os.path.dirname(p)) for p in image_paths]

print(f"Total images found: {len(image_paths)}")
print("Class distribution:", Counter(labels))

# Check for and remove corrupted images
corrupted = []
for p in tqdm(image_paths, desc="Checking images"):
    try:
        img = cv2.imread(p)
        if img is None:
            corrupted.append(p)
    except Exception:
        corrupted.append(p)

print(f"Corrupted images: {len(corrupted)}")
if corrupted:
    for p in corrupted:
        os.remove(p)
    image_paths = [p for p in image_paths if p not in corrupted]
    labels = [os.path.basename(os.path.dirname(p)) for p in image_paths]

# Train / val / test split  (70% / 15% / 15%)
X_train_paths, X_temp_paths, y_train, y_temp = train_test_split(
    image_paths, labels, test_size=0.3, stratify=labels, random_state=42
)
X_val_paths, X_test_paths, y_val, y_test = train_test_split(
    X_temp_paths, y_temp, test_size=0.5, stratify=y_temp, random_state=42
)

print(f"\nSplit sizes — Train: {len(X_train_paths)}, Val: {len(X_val_paths)}, Test: {len(X_test_paths)}")

# Persist splits for downstream scripts
with open('splits.pkl', 'wb') as f:
    pickle.dump((X_train_paths, X_val_paths, X_test_paths, y_train, y_val, y_test), f)

print("Splits saved to splits.pkl")