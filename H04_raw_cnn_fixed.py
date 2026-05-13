"""
H04_raw_cnn_fixed.py
─────────────────────
Retrains MobileNetV2 and EfficientNetB0 on RAW (uncropped) images
with CORRECT preprocessing — fixes the original training bug.

Original bug: both models trained with rescale=1./255 ([0,1])
  MobileNetV2  requires  [-1, 1]  via preprocess_input  (Sandler et al. 2019, Section 3.2)
  EfficientNetB0 requires [0, 255] raw — NO rescaling    (Tan & Le 2019, Section 4)

Reads  : splits.pkl              (from H02_data_validation.py)
Writes : mobilenetv2_raw.h5       (overwrites broken version)
         efficientnetb0_raw.h5    (overwrites broken version)
         mobilenetv2_raw_classification_report.txt
         mobilenetv2_raw_confusion_matrix.png
         mobilenetv2_raw_training_curves_phase1.png
         mobilenetv2_raw_training_curves_finetune.png
         efficientnetb0_raw_classification_report.txt
         efficientnetb0_raw_confusion_matrix.png
         efficientnetb0_raw_training_curves_phase1.png
         efficientnetb0_raw_training_curves_finetune.png
         model_results.json       (updated with corrected accuracies)
"""

import os
import json
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf

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
print("Loading splits.pkl ...")
with open('splits.pkl', 'rb') as f:
    X_train_paths, X_val_paths, X_test_paths, y_train, y_val, y_test = pickle.load(f)

train_df = pd.DataFrame({'filename': X_train_paths, 'class': y_train})
val_df   = pd.DataFrame({'filename': X_val_paths,   'class': y_val})
test_df  = pd.DataFrame({'filename': X_test_paths,  'class': y_test})

print(f"Train: {len(train_df):,}  |  Val: {len(val_df):,}  |  Test: {len(test_df):,}")


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
    y_prob = model.predict(test_gen, verbose=1)
    y_pred = np.argmax(y_prob, axis=1)
    y_true = test_gen.classes

    # Only report labels that are present in predictions or ground truth
    present = sorted(set(y_true) | set(y_pred))
    present_names = [class_names[i] for i in present if i < len(class_names)]

    report = classification_report(
        y_true, y_pred,
        labels=present,
        target_names=present_names,
        digits=4, zero_division=0
    )
    print(f"\n{tag} — Test Accuracy: {test_acc:.4f}")
    print(report)
    with open(f'{tag}_classification_report.txt', 'w') as f:
        f.write(f"Model: {tag}\nTest Accuracy: {test_acc:.4f}\n\n{report}")

    cm = confusion_matrix(y_true, y_pred, labels=present)
    fig, ax = plt.subplots(figsize=(16, 14))
    sns.heatmap(cm, annot=True, fmt='d',
                xticklabels=present_names, yticklabels=present_names,
                cmap='Blues', linewidths=0.4, ax=ax)
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
    print(f"Curves → {tag}_training_curves{suffix}.png")


def update_results(key, value):
    results = {}
    if os.path.exists('model_results.json'):
        with open('model_results.json') as f:
            results = json.load(f)
    results[key] = round(float(value), 4)
    with open('model_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"model_results.json updated → {key}: {value:.4f}")


# ════════════════════════════════════════════════════════════════
# A.  MobileNetV2 — CORRECT preprocessing: [-1, 1]
# ════════════════════════════════════════════════════════════════
print("\n" + "=" * 64)
print("  MobileNetV2 (raw)  |  preprocessing: [-1, 1]")
print("  mobilenet_v2.preprocess_input maps [0,255] → [-1, 1]")
print("  Ref: Sandler et al. 2019, Section 3.2")
print("=" * 64)

# Training: augmentation + correct preprocessing
mnv2_train_gen = ImageDataGenerator(
    preprocessing_function=mnv2_preprocess,   # ← correct: [-1, 1]
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    brightness_range=[0.8, 1.2],
    fill_mode='nearest',
).flow_from_dataframe(
    train_df, x_col='filename', y_col='class',
    target_size=(IMG_SIZE, IMG_SIZE), batch_size=BATCH_SIZE,
    class_mode='categorical'
)

mnv2_val_gen = ImageDataGenerator(
    preprocessing_function=mnv2_preprocess    # ← correct: [-1, 1]
).flow_from_dataframe(
    val_df, x_col='filename', y_col='class',
    target_size=(IMG_SIZE, IMG_SIZE), batch_size=BATCH_SIZE,
    class_mode='categorical'
)

mnv2_test_gen = ImageDataGenerator(
    preprocessing_function=mnv2_preprocess    # ← correct: [-1, 1]
).flow_from_dataframe(
    test_df, x_col='filename', y_col='class',
    target_size=(IMG_SIZE, IMG_SIZE), batch_size=BATCH_SIZE,
    class_mode='categorical', shuffle=False
)

num_classes = len(mnv2_train_gen.class_indices)
class_names = [k for k, _ in sorted(
    mnv2_train_gen.class_indices.items(), key=lambda x: x[1]
)]
print(f"Classes: {num_classes}  |  {class_names}")

# Save class names file
with open('class_names.txt', 'w') as f:
    for n in class_names: f.write(n + '\n')
print("Class names saved → class_names.txt")

# Build model — frozen backbone
base_mnv2 = MobileNetV2(weights='imagenet', include_top=False,
                         input_shape=(IMG_SIZE, IMG_SIZE, 3))
base_mnv2.trainable = False
x   = GlobalAveragePooling2D()(base_mnv2.output)
x   = Dense(256, activation='relu')(x)
x   = Dropout(0.4)(x)
x   = Dense(128, activation='relu')(x)
x   = Dropout(0.3)(x)
out = Dense(num_classes, activation='softmax')(x)
model_mnv2 = Model(base_mnv2.input, out)
model_mnv2.compile(optimizer=Adam(1e-3),
                   loss='categorical_crossentropy', metrics=['accuracy'])
model_mnv2.summary()

# Phase 1 — frozen base
print("\n--- MobileNetV2 Phase 1: Frozen base ---")
h1 = model_mnv2.fit(mnv2_train_gen, validation_data=mnv2_val_gen,
                     epochs=30, callbacks=get_callbacks())
save_curves(h1, 'mobilenetv2_raw', 'phase1')

# Phase 2 — fine-tune top 30 layers
print("\n--- MobileNetV2 Phase 2: Fine-tune top 30 layers ---")
base_mnv2.trainable = True
for layer in base_mnv2.layers[:-30]:
    layer.trainable = False
model_mnv2.compile(optimizer=Adam(1e-5),
                   loss='categorical_crossentropy', metrics=['accuracy'])
h2 = model_mnv2.fit(mnv2_train_gen, validation_data=mnv2_val_gen,
                     epochs=15, callbacks=get_callbacks())
save_curves(h2, 'mobilenetv2_raw', 'finetune')

# Evaluate
_, acc_mnv2 = model_mnv2.evaluate(mnv2_test_gen, verbose=0)
print(f"\nMobileNetV2 (raw, corrected) test accuracy: {acc_mnv2:.4f}")
save_eval_artefacts(model_mnv2, mnv2_test_gen, class_names,
                    'mobilenetv2_raw', acc_mnv2)
model_mnv2.save('mobilenetv2_raw.h5')
print("Model saved → mobilenetv2_raw.h5")
update_results('MobileNetV2 Raw', acc_mnv2)


# ════════════════════════════════════════════════════════════════
# B.  EfficientNetB0 — CORRECT preprocessing: raw [0, 255]
# ════════════════════════════════════════════════════════════════
print("\n" + "=" * 64)
print("  EfficientNetB0 (raw)  |  raw [0, 255] — NO rescaling")
print("  EfficientNet handles normalisation internally")
print("  Ref: Tan & Le 2019, Section 4")
print("=" * 64)

# Training: augmentation ONLY — no rescaling, no preprocessing_function
efn_train_gen = ImageDataGenerator(
    # No rescale, no preprocessing_function — raw [0, 255] ← correct for EfficientNet
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    brightness_range=[0.8, 1.2],
    fill_mode='nearest',
).flow_from_dataframe(
    train_df, x_col='filename', y_col='class',
    target_size=(IMG_SIZE, IMG_SIZE), batch_size=BATCH_SIZE,
    class_mode='categorical'
)

efn_val_gen = ImageDataGenerator(
    # No rescale — raw [0, 255]
).flow_from_dataframe(
    val_df, x_col='filename', y_col='class',
    target_size=(IMG_SIZE, IMG_SIZE), batch_size=BATCH_SIZE,
    class_mode='categorical'
)

efn_test_gen = ImageDataGenerator(
    # No rescale — raw [0, 255]
).flow_from_dataframe(
    test_df, x_col='filename', y_col='class',
    target_size=(IMG_SIZE, IMG_SIZE), batch_size=BATCH_SIZE,
    class_mode='categorical', shuffle=False
)

# Build model — frozen backbone
base_efn = EfficientNetB0(weights='imagenet', include_top=False,
                           input_shape=(IMG_SIZE, IMG_SIZE, 3))
base_efn.trainable = False
x   = GlobalAveragePooling2D()(base_efn.output)
x   = Dense(256, activation='relu')(x)
x   = Dropout(0.4)(x)
x   = Dense(128, activation='relu')(x)
x   = Dropout(0.3)(x)
out = Dense(num_classes, activation='softmax')(x)
model_efn = Model(base_efn.input, out)
model_efn.compile(optimizer=Adam(1e-3),
                  loss='categorical_crossentropy', metrics=['accuracy'])
model_efn.summary()

# Phase 1 — frozen base
print("\n--- EfficientNetB0 Phase 1: Frozen base ---")
h1 = model_efn.fit(efn_train_gen, validation_data=efn_val_gen,
                    epochs=30, callbacks=get_callbacks())
save_curves(h1, 'efficientnetb0_raw', 'phase1')

# Phase 2 — fine-tune top 30 layers
print("\n--- EfficientNetB0 Phase 2: Fine-tune top 30 layers ---")
base_efn.trainable = True
for layer in base_efn.layers[:-30]:
    layer.trainable = False
model_efn.compile(optimizer=Adam(1e-5),
                  loss='categorical_crossentropy', metrics=['accuracy'])
h2 = model_efn.fit(efn_train_gen, validation_data=efn_val_gen,
                    epochs=15, callbacks=get_callbacks())
save_curves(h2, 'efficientnetb0_raw', 'finetune')

# Evaluate
_, acc_efn = model_efn.evaluate(efn_test_gen, verbose=0)
print(f"\nEfficientNetB0 (raw, corrected) test accuracy: {acc_efn:.4f}")
save_eval_artefacts(model_efn, efn_test_gen, class_names,
                    'efficientnetb0_raw', acc_efn)
model_efn.save('efficientnetb0_raw.h5')
print("Model saved → efficientnetb0_raw.h5")
update_results('EfficientNetB0 Raw', acc_efn)


# ── Final summary ─────────────────────────────────────────────────────────────
print("\n" + "=" * 64)
print("  CORRECTED RAW MODEL RESULTS")
print("=" * 64)
print(f"  MobileNetV2  Raw (corrected) : {acc_mnv2:.4f}")
print(f"  EfficientNetB0 Raw (corrected): {acc_efn:.4f}")

with open('model_results.json') as f:
    all_results = json.load(f)

print("\n  Full results table:")
print(f"  {'Model':<30} {'Accuracy':>10}")
print("  " + "-" * 42)
for k, v in sorted(all_results.items(), key=lambda x: -x[1]):
    bar = '█' * int(v * 30)
    print(f"  {k:<30} : {v:.4f}  {bar}")

print("\nNext: run H07_compare_models.py to regenerate the comparison chart.")
