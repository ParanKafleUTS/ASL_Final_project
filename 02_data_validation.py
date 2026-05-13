import os
import cv2
import numpy as np
from imutils import paths
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from collections import Counter

target_dir = 'subset_5000'
IMG_SIZE = 224

image_paths = list(paths.list_images(target_dir))
labels = [os.path.basename(os.path.dirname(p)) for p in image_paths]

print("Class distribution:", Counter(labels))

# Check corrupted images
corrupted = []
for p in tqdm(image_paths):
    try:
        img = cv2.imread(p)
        if img is None:
            corrupted.append(p)
    except:
        corrupted.append(p)
print(f"Corrupted images: {len(corrupted)}")
if corrupted:
    for p in corrupted:
        os.remove(p)
    image_paths = [p for p in image_paths if p not in corrupted]
    labels = [os.path.basename(os.path.dirname(p)) for p in image_paths]

# Train/val/test split
X_train_paths, X_temp_paths, y_train, y_temp = train_test_split(
    image_paths, labels, test_size=0.3, stratify=labels, random_state=42
)
X_val_paths, X_test_paths, y_val, y_test = train_test_split(
    X_temp_paths, y_temp, test_size=0.5, stratify=y_temp, random_state=42
)

print(f"Train: {len(X_train_paths)}, Val: {len(X_val_paths)}, Test: {len(X_test_paths)}")

# Save splits for later use (optional)
import pickle
with open('splits.pkl', 'wb') as f:
    pickle.dump((X_train_paths, X_val_paths, X_test_paths, y_train, y_val, y_test), f)