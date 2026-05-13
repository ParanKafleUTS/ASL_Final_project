"""
H04_raw_cnn.py
──────────────
Step 4 — Train MobileNetV2 and EfficientNetB0 on raw (uncropped) images.

Reads  : splits.pkl
Writes : mobilenetv2_raw.h5
         efficientnetb0_raw.h5
         class_names.txt
         mobilenetv2_raw_classification_report.txt
         mobilenetv2_raw_confusion_matrix.png
         mobilenetv2_raw_training_curves_phase1.png
         mobilenetv2_raw_training_curves_finetune.png
         (same set for efficientnetb0_raw)
         model_results.json  (appended)

Academic improvements:
  1. MobileNetV2: preprocessing via mobilenet_v2.preprocess_input → [-1, 1].
     Ref: Sandler et al. 2019, Section 3.2.
  2. EfficientNetB0: raw [0, 255] — handled internally by the network.
     Ref: Tan & Le 2019, Section 4.
  3. Two-phase training: frozen base → fine-tune top-30 layers.
  4. Confusion matrix, per-class F1 report, training curves.
  5. Results appended to model_results.json.
"""

import os
import json
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2, EfficientNetB0
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mnv2_preprocess
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import classification_report, confusion_matrix

IMG_SIZE   = 224
BATCH_SIZE = 32

# ── Load splits ───────────────────────────────────────────────────────────────
with open('splits.pkl', 'rb') as f:
    X_train_paths, X_val_paths, X_test_paths, y_train, y_val, y_test = pickle.load(f)

train_df = pd.DataFrame({'filename': X_train_paths, 'class': y_train})
val_df   = pd.DataFrame({'filename': X_val_paths,   'class': y_val})
test_df  = pd.DataFrame({'filename': X_test_paths,  'class': y_test})

# ── Shared helpers ─────────────────────────────────────────────────────────────
def get_callbacks():
    return [
        EarlyStopping(monitor='val_loss', patience=5,
                      restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                          patience=2, min_lr=1e-7, verbose=1),
    ]

def save_eval_artefacts(model, test_gen, class_names, tag, test_acc):
    test_gen.reset()
    y_pred = np.argmax(model.predict(test_gen, verbose=1), axis=1)
    y_true = test_gen.classes
    report = classification_report(y_true, y_pred, target_names=class_names, digits=4)
    print(f"\n{tag} Classification Report:\n", report)
    with open(f'{tag}_classification_report.txt', 'w') as f:
        f.write(report)
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(16, 14))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_names,
                yticklabels=class_names, cmap='Blues', linewidths=0.4, ax=ax)
    ax.set_xlabel('Predicted'); ax.set_ylabel('True')
    ax.set_title(f'{tag} — Confusion Matrix (Acc: {test_acc:.4f})')
    plt.tight_layout()
    plt.savefig(f'{tag}_confusion_matrix.png', dpi=150); plt.close()
    print(f"Confusion matrix → {tag}_confusion_matrix.png")

def save_curves(history, tag, phase=''):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history.history['accuracy'],     label='Train', color='steelblue')
    axes[0].plot(history.history['val_accuracy'], label='Val',   color='darkorange')
    axes[0].set_title('Accuracy'); axes[0].legend(); axes[0].grid(alpha=0.3)
    axes[1].plot(history.history['loss'],         label='Train', color='steelblue')
    axes[1].plot(history.history['val_loss'],     label='Val',   color='darkorange')
    axes[1].set_title('Loss'); axes[1].legend(); axes[1].grid(alpha=0.3)
    suffix = f'_{phase}' if phase else ''
    fig.suptitle(f'{tag} Training Curves'); plt.tight_layout()
    plt.savefig(f'{tag}_training_curves{suffix}.png', dpi=150); plt.close()

def update_results(key, value):
    results = {}
    if os.path.exists('model_results.json'):
        with open('model_results.json') as f:
            results = json.load(f)
    results[key] = round(float(value), 4)
    with open('model_results.json', 'w') as f:
        json.dump(results, f, indent=2)

# ════════════════════════════════════════════════════════════════
# A.  MobileNetV2 — Raw images
# ════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  Training MobileNetV2 (raw)  |  preprocessing: [-1, 1]")
print("  Ref: Sandler et al. 2019, Section 3.2")
print("=" * 60)

# MobileNetV2 REQUIRES [-1, 1] — use the official preprocess_input function
aug_mnv2  = ImageDataGenerator(preprocessing_function=mnv2_preprocess,
                                rotation_range=15, width_shift_range=0.1,
                                height_shift_range=0.1, zoom_range=0.1,
                                brightness_range=[0.8, 1.2], fill_mode='nearest')
val_mnv2  = ImageDataGenerator(preprocessing_function=mnv2_preprocess)
test_mnv2 = ImageDataGenerator(preprocessing_function=mnv2_preprocess)

train_gen_mnv2 = aug_mnv2.flow_from_dataframe(
    train_df, x_col='filename', y_col='class',
    target_size=(IMG_SIZE, IMG_SIZE), batch_size=BATCH_SIZE, class_mode='categorical')
val_gen_mnv2 = val_mnv2.flow_from_dataframe(
    val_df, x_col='filename', y_col='class',
    target_size=(IMG_SIZE, IMG_SIZE), batch_size=BATCH_SIZE, class_mode='categorical')
test_gen_mnv2 = test_mnv2.flow_from_dataframe(
    test_df, x_col='filename', y_col='class',
    target_size=(IMG_SIZE, IMG_SIZE), batch_size=BATCH_SIZE, class_mode='categorical',
    shuffle=False)

num_classes = len(train_gen_mnv2.class_indices)
class_names = [k for k, _ in sorted(train_gen_mnv2.class_indices.items(), key=lambda x: x[1])]
print(f"Classes: {num_classes}")

# Save class names
with open('class_names.txt', 'w') as f:
    for n in class_names: f.write(n + '\n')
print("Class names saved → class_names.txt")

# Build model
base_mnv2 = MobileNetV2(weights='imagenet', include_top=False,
                          input_shape=(IMG_SIZE, IMG_SIZE, 3))
base_mnv2.trainable = False
x   = GlobalAveragePooling2D()(base_mnv2.output)
x   = Dense(256, activation='relu')(x); x = Dropout(0.4)(x)
x   = Dense(128, activation='relu')(x); x = Dropout(0.3)(x)
out = Dense(num_classes, activation='softmax')(x)
model_mnv2 = Model(base_mnv2.input, out)
model_mnv2.compile(optimizer=Adam(1e-3), loss='categorical_crossentropy', metrics=['accuracy'])

print("\n--- Phase 1: Frozen base ---")
h1 = model_mnv2.fit(train_gen_mnv2, validation_data=val_gen_mnv2,
                     epochs=30, callbacks=get_callbacks())
save_curves(h1, 'mobilenetv2_raw', 'phase1')

print("\n--- Phase 2: Fine-tune top 30 layers ---")
base_mnv2.trainable = True
for layer in base_mnv2.layers[:-30]:
    layer.trainable = False
model_mnv2.compile(optimizer=Adam(1e-5), loss='categorical_crossentropy', metrics=['accuracy'])
h2 = model_mnv2.fit(train_gen_mnv2, validation_data=val_gen_mnv2,
                     epochs=15, callbacks=get_callbacks())
save_curves(h2, 'mobilenetv2_raw', 'finetune')

_, acc_mnv2 = model_mnv2.evaluate(test_gen_mnv2, verbose=0)
print(f"\nMobileNetV2 (raw) test accuracy: {acc_mnv2:.4f}")
save_eval_artefacts(model_mnv2, test_gen_mnv2, class_names, 'mobilenetv2_raw', acc_mnv2)
model_mnv2.save('mobilenetv2_raw.h5')
print("Model saved → mobilenetv2_raw.h5")
update_results('MobileNetV2 Raw', acc_mnv2)


# ════════════════════════════════════════════════════════════════
# B.  EfficientNetB0 — Raw images
# ════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  Training EfficientNetB0 (raw)  |  raw [0, 255]")
print("  Ref: Tan & Le 2019, Section 4")
print("=" * 60)

# EfficientNetB0 expects raw [0, 255] — do NOT rescale
aug_efn  = ImageDataGenerator(rotation_range=15, width_shift_range=0.1,
                                height_shift_range=0.1, zoom_range=0.1,
                                brightness_range=[0.8, 1.2], fill_mode='nearest')
val_efn  = ImageDataGenerator()
test_efn = ImageDataGenerator()

train_gen_efn = aug_efn.flow_from_dataframe(
    train_df, x_col='filename', y_col='class',
    target_size=(IMG_SIZE, IMG_SIZE), batch_size=BATCH_SIZE, class_mode='categorical')
val_gen_efn = val_efn.flow_from_dataframe(
    val_df, x_col='filename', y_col='class',
    target_size=(IMG_SIZE, IMG_SIZE), batch_size=BATCH_SIZE, class_mode='categorical')
test_gen_efn = test_efn.flow_from_dataframe(
    test_df, x_col='filename', y_col='class',
    target_size=(IMG_SIZE, IMG_SIZE), batch_size=BATCH_SIZE, class_mode='categorical',
    shuffle=False)

base_efn = EfficientNetB0(weights='imagenet', include_top=False,
                           input_shape=(IMG_SIZE, IMG_SIZE, 3))
base_efn.trainable = False
x   = GlobalAveragePooling2D()(base_efn.output)
x   = Dense(256, activation='relu')(x); x = Dropout(0.4)(x)
x   = Dense(128, activation='relu')(x); x = Dropout(0.3)(x)
out = Dense(num_classes, activation='softmax')(x)
model_efn = Model(base_efn.input, out)
model_efn.compile(optimizer=Adam(1e-3), loss='categorical_crossentropy', metrics=['accuracy'])

print("\n--- Phase 1: Frozen base ---")
h1 = model_efn.fit(train_gen_efn, validation_data=val_gen_efn,
                    epochs=30, callbacks=get_callbacks())
save_curves(h1, 'efficientnetb0_raw', 'phase1')

print("\n--- Phase 2: Fine-tune top 30 layers ---")
base_efn.trainable = True
for layer in base_efn.layers[:-30]:
    layer.trainable = False
model_efn.compile(optimizer=Adam(1e-5), loss='categorical_crossentropy', metrics=['accuracy'])
h2 = model_efn.fit(train_gen_efn, validation_data=val_gen_efn,
                    epochs=15, callbacks=get_callbacks())
save_curves(h2, 'efficientnetb0_raw', 'finetune')

_, acc_efn = model_efn.evaluate(test_gen_efn, verbose=0)
print(f"\nEfficientNetB0 (raw) test accuracy: {acc_efn:.4f}")
save_eval_artefacts(model_efn, test_gen_efn, class_names, 'efficientnetb0_raw', acc_efn)
model_efn.save('efficientnetb0_raw.h5')
print("Model saved → efficientnetb0_raw.h5")
update_results('EfficientNetB0 Raw', acc_efn)

print("\nAll raw CNN results saved → model_results.json")
