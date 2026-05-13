import os
import cv2
import numpy as np
from imutils import paths
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from collections import Counter
import pickle

# Use full dataset path
extracted_root = 'ASL_Alphabet_Dataset'
train_folder = os.path.join(extracted_root, 'asl_alphabet_train')

IMG_SIZE = 224

image_paths = list(paths.list_images(train_folder))
labels = [os.path.basename(os.path.dirname(p)) for p in image_paths]

print("Class distribution:", Counter(labels))
print(f"Total images: {len(image_paths)}")

# Check corrupted images
corrupted = []
for p in tqdm(image_paths, desc="Checking for corrupted images"):
    try:
        img = cv2.imread(p)
        if img is None:
            corrupted.append(p)
    except:
        corrupted.append(p)

print(f"Corrupted images found: {len(corrupted)}")
if corrupted:
    for p in corrupted:
        os.remove(p)
    image_paths = [p for p in image_paths if p not in corrupted]
    labels = [os.path.basename(os.path.dirname(p)) for p in image_paths]
    print(f"Removed {len(corrupted)} corrupted images")

print("Updated class distribution:", Counter(labels))
print(f"Total images after cleanup: {len(image_paths)}")

# Train/val/test split (70/15/15)
X_train_paths, X_temp_paths, y_train, y_temp = train_test_split(
    image_paths, labels, test_size=0.3, stratify=labels, random_state=42
)
X_val_paths, X_test_paths, y_val, y_test = train_test_split(
    X_temp_paths, y_temp, test_size=0.5, stratify=y_temp, random_state=42
)

print(f"\nDataset Split:")
print(f"Train: {len(X_train_paths)} ({len(X_train_paths)/len(image_paths)*100:.1f}%)")
print(f"Val: {len(X_val_paths)} ({len(X_val_paths)/len(image_paths)*100:.1f}%)")
print(f"Test: {len(X_test_paths)} ({len(X_test_paths)/len(image_paths)*100:.1f}%)")

print(f"\nTrain class distribution: {Counter(y_train)}")
print(f"Val class distribution: {Counter(y_val)}")
print(f"Test class distribution: {Counter(y_test)}")

# Save splits for later use
with open('splits.pkl', 'wb') as f:
    pickle.dump((X_train_paths, X_val_paths, X_test_paths, y_train, y_val, y_test), f)
    
print("\nDataset splits saved to 'splits.pkl'")