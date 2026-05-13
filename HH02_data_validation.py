"""
H02_data_validation.py
──────────────────────
Step 2 — Load all images, check for corruption, produce a 70/15/15
stratified split, and save splits.pkl.

Reads  : train_folder.txt  (written by H01)
Writes : splits.pkl
         split_distribution.png
         split_stats.json

Split strategy follows DeepASLR (Page 2, Section A) and
Ma et al. (Sensors 2022, Page 5, Section 3.1).
"""

import os
import cv2
import json
import pickle
import numpy as np
import matplotlib.pyplot as plt
from imutils import paths
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from collections import Counter

# ── Read training folder written by H01 ──────────────────────────────────────
with open('train_folder.txt', 'r') as f:
    target_dir = f.read().strip()

IMG_SIZE = 224

print(f"Loading images from: {target_dir}")
image_paths = list(paths.list_images(target_dir))
labels      = [os.path.basename(os.path.dirname(p)) for p in image_paths]

print(f"Total images found : {len(image_paths):,}")
print(f"Class distribution : {dict(sorted(Counter(labels).items()))}")

# ── Remove corrupted images ───────────────────────────────────────────────────
corrupted = []
for p in tqdm(image_paths, desc="Checking images"):
    try:
        img = cv2.imread(p)
        if img is None:
            corrupted.append(p)
    except Exception:
        corrupted.append(p)

print(f"Corrupted images : {len(corrupted)}")
if corrupted:
    for p in corrupted:
        os.remove(p)
    image_paths = [p for p in image_paths if p not in corrupted]
    labels      = [os.path.basename(os.path.dirname(p)) for p in image_paths]
    print(f"Images after clean: {len(image_paths):,}")

# ── 70 / 15 / 15 stratified split ─────────────────────────────────────────────
X_train, X_temp, y_train, y_temp = train_test_split(
    image_paths, labels, test_size=0.30, stratify=labels, random_state=42
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=42
)

n = len(image_paths)
print(f"\nSplit sizes:")
print(f"  Train : {len(X_train):6,} ({len(X_train)/n*100:.1f}%)")
print(f"  Val   : {len(X_val):6,} ({len(X_val)/n*100:.1f}%)")
print(f"  Test  : {len(X_test):6,} ({len(X_test)/n*100:.1f}%)")

# ── Split distribution chart ───────────────────────────────────────────────────
sorted_classes = sorted(set(labels))
train_dist = Counter(y_train)
val_dist   = Counter(y_val)
test_dist  = Counter(y_test)

x = np.arange(len(sorted_classes))
w = 0.28
fig, ax = plt.subplots(figsize=(18, 5))
ax.bar(x - w, [train_dist[c] for c in sorted_classes], w, label='Train', color='steelblue')
ax.bar(x,     [val_dist[c]   for c in sorted_classes], w, label='Val',   color='darkorange')
ax.bar(x + w, [test_dist[c]  for c in sorted_classes], w, label='Test',  color='seagreen')
ax.set_xticks(x); ax.set_xticklabels(sorted_classes, fontsize=9)
ax.set_xlabel('Class'); ax.set_ylabel('Image Count')
ax.set_title('Class Distribution per Split (70 / 15 / 15)', fontsize=13)
ax.legend(); plt.tight_layout()
plt.savefig('split_distribution.png', dpi=150); plt.close()
print("\nSplit distribution chart saved → split_distribution.png")

# ── Save stats JSON ───────────────────────────────────────────────────────────
split_stats = {
    'total': n, 'train': len(X_train), 'val': len(X_val), 'test': len(X_test),
    'corrupted': len(corrupted), 'num_classes': len(sorted_classes),
    'train_ratio': round(len(X_train)/n, 4),
    'val_ratio'  : round(len(X_val)/n,   4),
    'test_ratio' : round(len(X_test)/n,  4),
}
with open('split_stats.json', 'w') as f:
    json.dump(split_stats, f, indent=2)
print("Split stats saved → split_stats.json")

# ── Persist splits ────────────────────────────────────────────────────────────
with open('splits.pkl', 'wb') as f:
    pickle.dump((X_train, X_val, X_test, y_train, y_val, y_test), f)
print("Splits saved → splits.pkl")
