import zipfile
import os
import random
import shutil

archive_path = 'archive.zip'
extract_path = '.'  # current folder

# Define expected structure after extraction
extracted_root = 'ASL_Alphabet_Dataset'
train_folder = os.path.join(extracted_root, 'asl_alphabet_train')

# Check if already extracted (by checking existence of train_folder)
if os.path.exists(train_folder):
    print("Dataset already extracted. Skipping extraction.")
else:
    print("Extracting archive...")
    with zipfile.ZipFile(archive_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)
    print("Extraction completed.")

# Verify that train_folder exists
if not os.path.exists(train_folder):
    raise FileNotFoundError(f"Training folder '{train_folder}' not found. Please check the archive structure.")

# Get list of letter classes (subdirectories inside train_folder)
letter_classes = [d for d in os.listdir(train_folder) if os.path.isdir(os.path.join(train_folder, d))]
print(f"Letter classes found: {letter_classes}")

# Target directory for balanced subset
target_dir = 'subset_5000'
os.makedirs(target_dir, exist_ok=True)

samples_per_class = 172

for letter in letter_classes:
    cls_source = os.path.join(train_folder, letter)
    
    # Get all image files in this class folder
    all_files = os.listdir(cls_source)
    image_files = [f for f in all_files if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
    
    if not image_files:
        print(f"Warning: No image files found in {cls_source}")
        continue
    
    # Select random samples (if fewer than needed, take all)
    num_to_select = min(samples_per_class, len(image_files))
    selected = random.sample(image_files, num_to_select)
    
    # Create class subfolder in target directory
    cls_target = os.path.join(target_dir, letter)
    os.makedirs(cls_target, exist_ok=True)
    
    # Copy selected images
    for img in selected:
        src_path = os.path.join(cls_source, img)
        dst_path = os.path.join(cls_target, img)
        shutil.copy(src_path, dst_path)
    
    print(f"Copied {num_to_select} images to {cls_target}")

print("Balanced subset created.")