"""
H_posthoc_eval.py
──────────────────
Generates ALL evaluation artefacts from ALREADY TRAINED .h5 models.
Run this to avoid full retraining — takes ~20 minutes on CPU.

Reads  : splits.pkl              (from H02)
         processed_splits.pkl    (from H05)
         landmark_cache.pkl      (from H03)
         landmark_label_encoder.pkl
         *.h5  model files        (from training scripts)
Writes : *_confusion_matrix.png
         *_classification_report.txt
         model_results.json
"""

import os
import json
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from collections import Counter
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess
from tensorflow.keras.applications.efficientnet import preprocess_input as efficientnet_preprocess

IMG_SIZE   = 224
BATCH_SIZE = 64
AUTOTUNE   = tf.data.AUTOTUNE


# ══════════════════════════════════════════════════════════════════════════════
# Shared helpers
# ══════════════════════════════════════════════════════════════════════════════

def save_eval_artefacts(y_true, y_pred, class_names, tag, test_acc):
    present_labels = sorted(set(y_true) | set(y_pred))
    present_names  = [class_names[i] for i in present_labels if i < len(class_names)]

    report = classification_report(
        y_true, y_pred,
        labels=present_labels,
        target_names=present_names,
        digits=4,
        zero_division=0,
    )
    print(f"\n{tag} — Test Accuracy: {test_acc:.4f}")
    print(report)
    with open(f'{tag}_classification_report.txt', 'w') as f:
        f.write(f"Model: {tag}\nTest Accuracy: {test_acc:.4f}\n\n{report}")
    print(f"  Report → {tag}_classification_report.txt")

    cm = confusion_matrix(y_true, y_pred, labels=present_labels)
    fig, ax = plt.subplots(figsize=(16, 14))
    sns.heatmap(cm, annot=True, fmt='d',
                xticklabels=present_names, yticklabels=present_names,
                cmap='Blues', linewidths=0.4, ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title(f'{tag} — Confusion Matrix  (Acc: {test_acc:.4f})')
    plt.tight_layout()
    plt.savefig(f'{tag}_confusion_matrix.png', dpi=150)
    plt.close()
    print(f"  Matrix  → {tag}_confusion_matrix.png")


def update_results(key, value):
    results = {}
    if os.path.exists('model_results.json'):
        with open('model_results.json') as f:
            results = json.load(f)
    results[key] = round(float(value), 4)
    with open('model_results.json', 'w') as f:
        json.dump(results, f, indent=2)


def load_model(filename):
    if not os.path.exists(filename):
        print(f"  [SKIP] {filename} not found.")
        return None
    print(f"\nLoading {filename}...")
    return tf.keras.models.load_model(filename, compile=False)


def predict_ds(model, test_ds):
    y_true_list, y_pred_list = [], []
    for imgs, labels in test_ds:
        preds = model.predict(imgs, verbose=0)
        y_pred_list.extend(np.argmax(preds, axis=1))
        y_true_list.extend(labels.numpy())
    return np.array(y_true_list), np.array(y_pred_list)


# ══════════════════════════════════════════════════════════════════════════════
# Dataset builders
# ══════════════════════════════════════════════════════════════════════════════

def process_path_mobilenet(file_path, label):
    img = tf.io.read_file(file_path)
    img = tf.image.decode_image(img, channels=3, expand_animations=False)
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
    img = mobilenet_preprocess(tf.cast(img, tf.float32))   # → [-1, 1]
    return img, label


def process_path_efficientnet(file_path, label):
    img = tf.io.read_file(file_path)
    img = tf.image.decode_image(img, channels=3, expand_animations=False)
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
    img = efficientnet_preprocess(tf.cast(img, tf.float32))  # → [0, 1]
    return img, label


def process_path_no_preprocess(file_path, label):
    """Use when the model has preprocessing baked in as an internal layer."""
    img = tf.io.read_file(file_path)
    img = tf.image.decode_image(img, channels=3, expand_animations=False)
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
    return tf.cast(img, tf.float32), label   # → [0, 255]


def make_image_ds(paths, labels, backbone='mobilenet'):
    """Build a dataset from file paths. backbone: 'mobilenet' | 'efficientnet' | 'none'"""
    if backbone == 'mobilenet':
        fn = process_path_mobilenet
    elif backbone == 'efficientnet':
        fn = process_path_efficientnet
    else:
        fn = process_path_no_preprocess
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    ds = ds.map(fn, num_parallel_calls=AUTOTUNE)
    return ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)


def make_array_ds(arrays, labels, backbone='mobilenet'):
    """Build a dataset from numpy pixel arrays. backbone: 'mobilenet' | 'efficientnet' | 'none'"""
    imgs = tf.image.resize(tf.cast(arrays, tf.float32), [IMG_SIZE, IMG_SIZE])
    if backbone == 'mobilenet':
        imgs = mobilenet_preprocess(imgs)
    elif backbone == 'efficientnet':
        imgs = efficientnet_preprocess(imgs)
    # 'none' → leave as [0, 255]
    ds = tf.data.Dataset.from_tensor_slices((imgs, labels))
    return ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)


# ══════════════════════════════════════════════════════════════════════════════
# Preprocessing detection
# ══════════════════════════════════════════════════════════════════════════════

def has_builtin_preprocessing(model):
    """
    Returns True if the model contains a preprocessing layer in its first 4 layers.
    Covers Lambda, Rescaling, Normalization layers and any layer named 'preprocess*'.
    """
    preprocess_types = ('Lambda', 'Rescaling', 'Normalization')
    for layer in model.layers[:4]:
        if layer.__class__.__name__ in preprocess_types:
            return True
        if 'preprocess' in layer.name.lower():
            return True
    return False


def best_backbone_for(model, default_backbone):
    """
    If the model has preprocessing baked in, return 'none' so we don't double-apply.
    Otherwise return the supplied default_backbone string.
    """
    if has_builtin_preprocessing(model):
        print(f"  ↳ Internal preprocessing layer detected — skipping external preprocess_input")
        return 'none'
    return default_backbone


# ══════════════════════════════════════════════════════════════════════════════
# 1.  Landmark MLP
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60 + "\n  Landmark MLP\n" + "="*60)

if os.path.exists('landmark_cache.pkl') and os.path.exists('landmark_label_encoder.pkl'):
    with open('landmark_label_encoder.pkl', 'rb') as f:
        le = pickle.load(f)
    with open('landmark_cache.pkl', 'rb') as f:
        landmark_data, valid_labels, saved_classes = pickle.load(f)

    le.classes_ = saved_classes
    y_encoded = le.transform(valid_labels)

    _, X_temp, _, y_temp = train_test_split(
        landmark_data, y_encoded, test_size=0.30, stratify=y_encoded, random_state=42)
    _, X_lm_test, _, y_lm_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42)

    model = load_model('landmark_mlp.h5')
    if model:
        y_prob   = model.predict(np.array(X_lm_test), verbose=1, batch_size=128)
        y_pred   = np.argmax(y_prob, axis=1)
        test_acc = float(np.mean(y_pred == y_lm_test))
        save_eval_artefacts(y_lm_test, y_pred, list(le.classes_), 'landmark_mlp', test_acc)
        update_results('Landmark MLP', test_acc)
        del model; tf.keras.backend.clear_session()
else:
    print("  [SKIP] landmark_cache.pkl or encoder not found — run H03 first.")


# ══════════════════════════════════════════════════════════════════════════════
# 2.  Raw CNN models  (mobilenetv2_raw, efficientnetb0_raw)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60 + "\n  Raw CNN models\n" + "="*60)

with open('splits.pkl', 'rb') as f:
    _, _, X_test_raw, _, _, y_test_raw = pickle.load(f)

# Always load class names from file saved at training time for guaranteed index alignment
if os.path.exists('class_names.txt'):
    with open('class_names.txt') as f:
        unique_raw = [line.strip() for line in f if line.strip()]
    print(f"  Loaded {len(unique_raw)} class names from class_names.txt")
else:
    unique_raw = sorted(set(y_test_raw))
    print("  WARNING: class_names.txt not found — deriving from test split (may misalign).")
    with open('class_names.txt', 'w') as f:
        for n in unique_raw:
            f.write(n + '\n')

idx_raw      = {cls: i for i, cls in enumerate(unique_raw)}
y_test_raw_i = [idx_raw[y] for y in y_test_raw]

# MobileNetV2 Raw
model = load_model('mobilenetv2_raw.h5')
if model:
    backbone = best_backbone_for(model, 'mobilenet')
    test_ds  = make_image_ds(X_test_raw, y_test_raw_i, backbone=backbone)
    y_true, y_pred = predict_ds(model, test_ds)
    test_acc = float(np.mean(y_true == y_pred))
    save_eval_artefacts(y_true, y_pred, unique_raw, 'mobilenetv2_raw', test_acc)
    update_results('MobileNetV2 Raw', test_acc)
    del model; tf.keras.backend.clear_session()

# EfficientNetB0 Raw
model = load_model('efficientnetb0_raw.h5')
if model:
    backbone = best_backbone_for(model, 'efficientnet')
    test_ds  = make_image_ds(X_test_raw, y_test_raw_i, backbone=backbone)
    y_true, y_pred = predict_ds(model, test_ds)
    test_acc = float(np.mean(y_true == y_pred))
    save_eval_artefacts(y_true, y_pred, unique_raw, 'efficientnetb0_raw', test_acc)
    update_results('EfficientNetB0 Raw', test_acc)
    del model; tf.keras.backend.clear_session()


# ══════════════════════════════════════════════════════════════════════════════
# 3.  Cropped CNN models  (mobilenetv2_crop, efficientnetb0_crop)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60 + "\n  Cropped CNN models\n" + "="*60)

if os.path.exists('processed_splits.pkl'):
    with open('processed_splits.pkl', 'rb') as f:
        _, _, X_test_crop, _, _, y_test_crop = pickle.load(f)

    unique_crop   = sorted(set(y_test_crop))
    idx_crop      = {cls: i for i, cls in enumerate(unique_crop)}
    y_test_crop_i = np.array([idx_crop[y] for y in y_test_crop])

    # Detect whether the split holds file-path strings or numpy pixel arrays
    first_item     = X_test_crop[0] if not isinstance(X_test_crop, np.ndarray) else X_test_crop[0]
    crop_is_arrays = isinstance(first_item, np.ndarray)
    print(f"  processed_splits: {'numpy arrays' if crop_is_arrays else 'file paths'} detected")

    def build_crop_ds(model, default_backbone):
        backbone = best_backbone_for(model, default_backbone)
        if crop_is_arrays:
            return make_array_ds(np.array(X_test_crop), y_test_crop_i, backbone=backbone)
        else:
            return make_image_ds(X_test_crop, y_test_crop_i, backbone=backbone)

    # MobileNetV2 Crop
    model = load_model('mobilenetv2_crop.h5')
    if model:
        test_ds = build_crop_ds(model, 'mobilenet')
        y_true, y_pred = predict_ds(model, test_ds)
        test_acc = float(np.mean(y_true == y_pred))
        save_eval_artefacts(y_true, y_pred, unique_crop, 'mobilenetv2_crop', test_acc)
        update_results('MobileNetV2 Crop', test_acc)
        del model; tf.keras.backend.clear_session()

    # EfficientNetB0 Crop
    model = load_model('efficientnetb0_crop.h5')
    if model:
        test_ds = build_crop_ds(model, 'efficientnet')
        y_true, y_pred = predict_ds(model, test_ds)
        test_acc = float(np.mean(y_true == y_pred))
        save_eval_artefacts(y_true, y_pred, unique_crop, 'efficientnetb0_crop', test_acc)
        update_results('EfficientNetB0 Crop', test_acc)
        del model; tf.keras.backend.clear_session()
else:
    print("  [SKIP] processed_splits.pkl not found — run H05 first.")


# ══════════════════════════════════════════════════════════════════════════════
# 4.  Skeleton CNN models
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60 + "\n  Skeleton CNN models\n" + "="*60)

SKELETON_DIR = 'skeleton_images'
if os.path.isdir(SKELETON_DIR):
    from imutils import paths as imp_paths

    skel_all    = list(imp_paths.list_images(SKELETON_DIR))
    skel_labels = [os.path.basename(os.path.dirname(p)) for p in skel_all]
    unique_skel = sorted(set(skel_labels))

    _, temp_p, _, temp_l = train_test_split(
        skel_all, skel_labels, test_size=0.30, stratify=skel_labels, random_state=42)

    # Only stratify the second split if every class has >= 2 samples in the pool
    temp_counts  = Counter(temp_l)
    can_stratify = all(v >= 2 for v in temp_counts.values())
    if not can_stratify:
        print("  WARNING: some skeleton classes have <2 samples in the 30% pool "
              "— falling back to non-stratified split.")

    val_ratio = 0.15 / 0.30
    _, X_test_skel, _, y_test_skel = train_test_split(
        temp_p, temp_l,
        test_size=(1 - val_ratio),
        stratify=temp_l if can_stratify else None,
        random_state=42,
    )

    skel_idx = {cls: i for i, cls in enumerate(unique_skel)}
    y_skel_i = [skel_idx[y] for y in y_test_skel]

    # MobileNetV2 Skeleton
    model = load_model('mobilenetv2_skeleton.h5')
    if model:
        backbone = best_backbone_for(model, 'mobilenet')
        test_ds  = make_image_ds(X_test_skel, y_skel_i, backbone=backbone)
        y_true, y_pred = predict_ds(model, test_ds)
        test_acc = float(np.mean(y_true == y_pred))
        save_eval_artefacts(y_true, y_pred, unique_skel, 'mobilenetv2_skeleton', test_acc)
        update_results('MobileNetV2 Skeleton', test_acc)
        del model; tf.keras.backend.clear_session()

    # EfficientNetB0 Skeleton
    model = load_model('efficientnetb0_skeleton.h5')
    if model:
        backbone = best_backbone_for(model, 'efficientnet')
        test_ds  = make_image_ds(X_test_skel, y_skel_i, backbone=backbone)
        y_true, y_pred = predict_ds(model, test_ds)
        test_acc = float(np.mean(y_true == y_pred))
        save_eval_artefacts(y_true, y_pred, unique_skel, 'efficientnetb0_skeleton', test_acc)
        update_results('EfficientNetB0 Skeleton', test_acc)
        del model; tf.keras.backend.clear_session()
else:
    print("  [SKIP] skeleton_images/ not found — run Hgenerate first.")


# ══════════════════════════════════════════════════════════════════════════════
# 5.  Summary
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("  FINAL RESULTS SUMMARY")
print("="*60)
if os.path.exists('model_results.json'):
    with open('model_results.json') as f:
        results = json.load(f)
    for k, v in sorted(results.items(), key=lambda x: -x[1]):
        bar = '█' * int(v * 30)
        print(f"  {k:<30} : {v:.4f}  {bar}")
    best = max(results, key=results.get)
    print(f"\n  Best model : {best}  ({results[best]:.4f})")

print("\nNext steps:")
print("  python H07_compare_models.py    → comparison chart + ablation table")
print("  python H09_gradcam.py           → Grad-CAM heatmaps")
print("  python H10_latency_benchmark.py → inference latency")
print("  python H11_tsne.py              → t-SNE feature embeddings")
