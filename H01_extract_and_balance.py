import zipfile
import os

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
archive_path = 'archive (2).zip'
extract_path = 'ASL_Alphabet_Dataset'

# ---------------------------------------------------------------------------
# Step 1: Extract the archive (skip if already done)
# ---------------------------------------------------------------------------
if not os.path.exists(extract_path):
    print("Extracting archive...")
    with zipfile.ZipFile(archive_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)
    print("Extraction completed.")
else:
    print("Dataset already extracted. Skipping extraction.")

# ---------------------------------------------------------------------------
# Step 2: Locate the training folder.
#
# The dataset zip uses "asl_alphabhet_train" (note the double-h typo).
# We try that name first, then fall back to the correctly-spelled variant so
# the script works with either version of the archive.
#
# Expected structures (outer → inner nesting):
#   ASL_Alphabet_Dataset/asl_alphabhet_train/asl_alphabhet_train/A/ B/ ...
#   ASL_Alphabet_Dataset/asl_alphabet_train/asl_alphabet_train/A/ B/ ...
# ---------------------------------------------------------------------------
def _find_folder(base, *candidates):
    """Return the first existing path from a list of candidate sub-paths."""
    for candidate in candidates:
        p = os.path.join(base, *candidate) if isinstance(candidate, (list, tuple)) else os.path.join(base, candidate)
        if os.path.exists(p):
            return p
    return None

train_folder = _find_folder(
    extract_path,
    # typo variant (double-h) — nested
    ('asl_alphabhet_train', 'asl_alphabhet_train'),
    # typo variant — flat
    'asl_alphabhet_train',
    # correctly-spelled variant — nested
    ('asl_alphabet_train', 'asl_alphabet_train'),
    # correctly-spelled variant — flat
    'asl_alphabet_train',
)

if train_folder is None:
    raise FileNotFoundError(
        f"Training folder not found under '{extract_path}'. "
        "Expected 'asl_alphabhet_train' or 'asl_alphabet_train'. "
        "Please verify the archive structure."
    )

# ---------------------------------------------------------------------------
# Step 3: Locate the test folder (flat: one image per class).
# ---------------------------------------------------------------------------
test_folder = _find_folder(
    extract_path,
    ('asl_alphabhet_test', 'asl_alphabhet_test'),
    'asl_alphabhet_test',
    ('asl_alphabet_test', 'asl_alphabet_test'),
    'asl_alphabet_test',
)

# ---------------------------------------------------------------------------
# Step 4: Report class counts for the training set
# ---------------------------------------------------------------------------
letter_classes = [
    d for d in os.listdir(train_folder)
    if os.path.isdir(os.path.join(train_folder, d))
]
print(f"Found {len(letter_classes)} classes: {sorted(letter_classes)}")

total_images = 0
for cls in sorted(letter_classes):
    cls_path = os.path.join(train_folder, cls)
    images = [
        f for f in os.listdir(cls_path)
        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))
    ]
    total_images += len(images)
    print(f"  {cls}: {len(images)} images")

print(f"\nTotal training images: {total_images}")
print(f"Training folder ready at: {train_folder}")

# ---------------------------------------------------------------------------
# Step 5: Persist folder paths for downstream scripts
# ---------------------------------------------------------------------------
with open('train_folder.txt', 'w') as f:
    f.write(train_folder)
print("Training folder path saved to train_folder.txt")

if test_folder:
    with open('test_folder.txt', 'w') as f:
        f.write(test_folder)
    print(f"Test folder path saved to test_folder.txt  ({test_folder})")
else:
    print("Note: test folder not found — test_folder.txt not written.")