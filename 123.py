# -*- coding: utf-8 -*-
"""
Complete ASL Alphabet Recognition Pipeline - Robust Path Finding
"""

# ================================
# 0. Install required packages (uncomment if needed)
# ================================
# !pip install tensorflow keras-tuner opencv-python imutils pandas matplotlib tqdm mediapipe

import os
import zipfile
import cv2
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from imutils import paths
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.mixed_precision import set_global_policy
import keras_tuner as kt

# Enable mixed precision for faster GPU training
set_global_policy('mixed_float16')
print("Mixed precision enabled.")

# ================================
# 1. Extract dataset with robust path finding
# ================================
archive_path = 'archive (2).zip'   # Adjust name if different
extract_dir = 'asl_data'

if not os.path.exists(extract_dir):
    print(f"Extracting {archive_path} ...")
    with zipfile.ZipFile(archive_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)
    print("Extraction completed.")
else:
    print(f"Data already extracted in '{extract_dir}'. Skipping extraction.")

# Function to find the training folder (contains class subfolders like 'A', 'B', ...)
def find_train_folder(root_dir):
    print(f"\nSearching for training folder under: {root_dir}")
    for root, dirs, files in os.walk(root_dir):
        # Look for a folder that contains subdirectories named 'A', 'B', ... or contains 'train' in name
        if any(name in dirs for name in ['A', 'B', 'C', 'del', 'space', 'nothing']):
            print(f"Found training folder at: {root}")
            return root
        # Also accept any folder with 'train' in its name that is not empty
        if 'train' in root.lower() and len(dirs) > 10:  # heuristic for class folders
            print(f"Found potential training folder at: {root}")
            return root
    # If nothing found, list the contents of extract_dir for debugging
    print("\nCould not locate training folder. Directory tree:")
    for r, d, f in os.walk(extract_dir):
        level = r.replace(extract_dir, '').count(os.sep)
        indent = ' ' * 2 * level
        print(f"{indent}{os.path.basename(r)}/")
        if level < 2:  # show only top two levels
            subindent = ' ' * 2 * (level+1)
            for subd in d[:5]:
                print(f"{subindent}{subd}/")
    raise FileNotFoundError("Training folder not found. Please check the zip contents.")

train_root = find_train_folder(extract_dir)
print(f"Using training data from: {train_root}")

# Get all image paths and labels
image_paths = list(paths.list_images(train_root))
labels = [os.path.basename(os.path.dirname(p)) for p in image_paths]

print(f"Total images found: {len(image_paths)}")
print(f"Unique classes: {set(labels)}")

# Remove corrupted images (optional but safe)
corrupted = []
for p in tqdm(image_paths, desc="Checking images"):
    try:
        img = cv2.imread(p)
        if img is None:
            corrupted.append(p)
    except:
        corrupted.append(p)

if corrupted:
    print(f"Removing {len(corrupted)} corrupted images.")
    for p in corrupted:
        os.remove(p)
    # Update lists
    image_paths = [p for p in image_paths if p not in corrupted]
    labels = [os.path.basename(os.path.dirname(p)) for p in image_paths]

# ================================
# 2. Train/Val/Test split
# ================================
X_train, X_temp, y_train, y_temp = train_test_split(
    image_paths, labels, test_size=0.2, stratify=labels, random_state=42
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
)

print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

# Save splits for later use
with open('full_splits.pkl', 'wb') as f:
    pickle.dump((X_train, X_val, X_test, y_train, y_val, y_test), f)

# ================================
# 3. Data augmentation with noise & lighting
# ================================
IMG_SIZE = 128   # Reduced for speed, still sufficient for hand signs
BATCH_SIZE = 128 # Adjust based on GPU memory

def add_noise(img):
    """Add random Gaussian noise to simulate poor camera quality."""
    if np.random.rand() < 0.5:
        noise = np.random.normal(0, 0.05, img.shape)
        img = img + noise
        img = np.clip(img, 0., 1.)
    return img

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    brightness_range=[0.8, 1.2],
    channel_shift_range=20,
    fill_mode='nearest',
    preprocessing_function=add_noise
)

# Validation/test only rescaling
val_test_datagen = ImageDataGenerator(rescale=1./255)

# Create dataframes for generators
train_df = pd.DataFrame({'filename': X_train, 'class': y_train})
val_df = pd.DataFrame({'filename': X_val, 'class': y_val})
test_df = pd.DataFrame({'filename': X_test, 'class': y_test})

train_generator = train_datagen.flow_from_dataframe(
    train_df, x_col='filename', y_col='class',
    target_size=(IMG_SIZE, IMG_SIZE), batch_size=BATCH_SIZE,
    class_mode='categorical', shuffle=True
)

val_generator = val_test_datagen.flow_from_dataframe(
    val_df, x_col='filename', y_col='class',
    target_size=(IMG_SIZE, IMG_SIZE), batch_size=BATCH_SIZE,
    class_mode='categorical', shuffle=False
)

test_generator = val_test_datagen.flow_from_dataframe(
    test_df, x_col='filename', y_col='class',
    target_size=(IMG_SIZE, IMG_SIZE), batch_size=BATCH_SIZE,
    class_mode='categorical', shuffle=False
)

num_classes = len(train_generator.class_indices)
class_names = list(train_generator.class_indices.keys())
print(f"Number of classes: {num_classes}")
print(f"Class names: {class_names}")

# Save class names
with open('class_names.txt', 'w') as f:
    for name in class_names:
        f.write(f"{name}\n")

# ================================
# 4. Model building function with hyperparameters
# ================================
def build_model(hp):
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
    base_model.trainable = False
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    dropout_rate = hp.Float('dropout', 0.2, 0.5, step=0.1)
    dense_units = hp.Int('dense_units', 64, 256, step=64)
    x = Dense(dense_units, activation='relu')(x)
    x = Dropout(dropout_rate)(x)
    output = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=output)
    learning_rate = hp.Choice('learning_rate', [1e-3, 5e-4, 1e-4])
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

# ================================
# 5. Hyperparameter tuning with Keras Tuner
# ================================
tuner = kt.RandomSearch(
    build_model,
    objective='val_accuracy',
    max_trials=10,
    executions_per_trial=1,
    directory='kt_dir',
    project_name='asl_tuning'
)

tuner_callbacks = [EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)]

print("Starting hyperparameter search...")
tuner.search(
    train_generator,
    validation_data=val_generator,
    epochs=10,
    callbacks=tuner_callbacks,
    verbose=1
)

best_hps = tuner.get_best_hyperparameters()[0]
print(f"Best dropout: {best_hps.get('dropout')}")
print(f"Best dense units: {best_hps.get('dense_units')}")
print(f"Best learning rate: {best_hps.get('learning_rate')}")

best_model = tuner.get_best_models()[0]

# ================================
# 6. Train final model with early stopping and checkpoint
# ================================
# Unfreeze last 20 layers for fine-tuning
for layer in best_model.layers[0].layers[-20:]:
    layer.trainable = True

best_model.compile(
    optimizer=Adam(learning_rate=best_hps.get('learning_rate') / 10),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

callbacks = [
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
    ModelCheckpoint('best_asl_model.h5', monitor='val_accuracy', save_best_only=True, mode='max')
]

print("Training final model...")
history = best_model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=50,
    callbacks=callbacks,
    verbose=1
)

# ================================
# 7. Evaluate on test set
# ================================
test_loss, test_acc = best_model.evaluate(test_generator, verbose=1)
print(f"Test accuracy: {test_acc:.4f}")

best_model.save('asl_final_model.h5')

# Plot training curves
plt.figure(figsize=(12,4))
plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.title('Accuracy')
plt.legend()
plt.subplot(1,2,2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Loss')
plt.legend()
plt.savefig('training_curves.png')
plt.show()

print("Training complete. Best model saved as 'asl_final_model.h5'")
print("Class names saved in 'class_names.txt'")