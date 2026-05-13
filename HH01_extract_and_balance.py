"""
H01_extract_and_balance.py
──────────────────────────
Step 1 — Extract the ASL Alphabet archive and validate the dataset.

Uses the same relative-path convention as the original code:
  - Reads  : archive (2).zip  (current working directory)
  - Writes : ASL_Alphabet_Dataset/  (extracted archive)
             train_folder.txt       (path for downstream scripts)
             test_folder.txt        (optional)
             class_distribution.png
             dataset_stats.json

Dataset: Akash (2018) Kaggle ASL Alphabet — 87,000 images, 29 classes,
         3,000 images per class.  Hussain et al. (2022, CMC) Section 4.1.1.
"""

import zipfile
import os
import json
import matplotlib.pyplot as plt

# ── Configuration ──────────────────────────────────────────────────────────────
ARCHIVE_PATH = 'archive (2).zip'
EXTRACT_PATH = 'ASL_Alphabet_Dataset'
IMAGE_EXTS   = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')

# ── Step 1: Extract archive ────────────────────────────────────────────────────
if not os.path.exists(EXTRACT_PATH):
    print("Extracting archive...")
    with zipfile.ZipFile(ARCHIVE_PATH, 'r') as z:
        z.extractall(EXTRACT_PATH)
    print("Extraction complete.")
else:
    print("Dataset already extracted — skipping.")

# ── Step 2: Locate training folder ────────────────────────────────────────────
def _find_folder(base, *candidates):
    for c in candidates:
        p = os.path.join(base, *c) if isinstance(c, (list, tuple)) else os.path.join(base, c)
        if os.path.exists(p):
            return p
    return None

train_folder = _find_folder(
    EXTRACT_PATH,
    ('asl_alphabhet_train', 'asl_alphabhet_train'),
    'asl_alphabhet_train',
    ('asl_alphabet_train',  'asl_alphabet_train'),
    'asl_alphabet_train',
)
if train_folder is None:
    raise FileNotFoundError(
        f"Training folder not found under '{EXTRACT_PATH}'. "
        "Expected 'asl_alphabhet_train' or 'asl_alphabet_train'."
    )

test_folder = _find_folder(
    EXTRACT_PATH,
    ('asl_alphabhet_test', 'asl_alphabhet_test'), 'asl_alphabhet_test',
    ('asl_alphabet_test',  'asl_alphabet_test'),  'asl_alphabet_test',
)

# ── Step 3: Count images per class ────────────────────────────────────────────
letter_classes = sorted([
    d for d in os.listdir(train_folder)
    if os.path.isdir(os.path.join(train_folder, d))
])
print(f"\nFound {len(letter_classes)} classes: {letter_classes}")

class_counts = {}
total_images = 0
for cls in letter_classes:
    imgs = [f for f in os.listdir(os.path.join(train_folder, cls))
            if f.lower().endswith(IMAGE_EXTS)]
    class_counts[cls] = len(imgs)
    total_images += len(imgs)
    print(f"  {cls:10s}: {len(imgs):5d} images")

mean_val = total_images / len(letter_classes)
print(f"\nTotal  : {total_images:,}")
print(f"Mean   : {mean_val:.0f}")
print(f"Min    : {min(class_counts.values())}")
print(f"Max    : {max(class_counts.values())}")

# ── Step 4: Class distribution chart ─────────────────────────────────────────
sorted_cls    = sorted(class_counts.keys())
sorted_counts = [class_counts[c] for c in sorted_cls]

plt.figure(figsize=(18, 5))
bars = plt.bar(sorted_cls, sorted_counts, color='steelblue', edgecolor='white')
for bar, cnt in zip(bars, sorted_counts):
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 15,
             str(cnt), ha='center', va='bottom', fontsize=7)
plt.axhline(mean_val, color='tomato', linestyle='--', linewidth=1.2,
            label=f'Mean = {mean_val:.0f}')
plt.title('ASL Alphabet Dataset — Class Distribution', fontsize=13)
plt.xlabel('Class'); plt.ylabel('Image Count')
plt.legend(); plt.tight_layout()
plt.savefig('class_distribution.png', dpi=150); plt.close()
print("\nClass distribution chart saved → class_distribution.png")

# ── Step 5: Save stats JSON ───────────────────────────────────────────────────
stats = {
    'total_images'    : total_images,
    'num_classes'     : len(letter_classes),
    'classes'         : letter_classes,
    'counts_per_class': class_counts,
    'mean_per_class'  : round(mean_val, 1),
    'min_per_class'   : min(class_counts.values()),
    'max_per_class'   : max(class_counts.values()),
}
with open('dataset_stats.json', 'w') as f:
    json.dump(stats, f, indent=2)
print("Dataset stats saved → dataset_stats.json")

# ── Step 6: Persist folder paths for downstream scripts ───────────────────────
with open('train_folder.txt', 'w') as f:
    f.write(train_folder)
print(f"Training folder path saved → train_folder.txt  ({train_folder})")

if test_folder:
    with open('test_folder.txt', 'w') as f:
        f.write(test_folder)
    print(f"Test folder path saved    → test_folder.txt   ({test_folder})")
else:
    print("Note: test folder not found — test_folder.txt not written.")
