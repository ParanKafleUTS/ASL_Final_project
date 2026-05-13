"""
H03_landmark_mlp.py
────────────────────
Step 3 — Train a Multilayer Perceptron on MediaPipe hand landmarks.

Reads  : splits.pkl
Writes : landmark_cache.pkl
         landmark_mlp.h5
         landmark_label_encoder.pkl
         landmark_mlp_classification_report.txt
         landmark_mlp_confusion_matrix.png
         landmark_mlp_training_curves.png
         model_results.json  (appended)

Academic improvements:
  1. Wrist-relative coordinate normalisation for signer-independence.
     Hussain et al. (2022, CMC) Pages 3–4, Section 3.2.
  2. BatchNormalization for training stability.
  3. Confusion matrix, per-class F1 report, training curves.
  4. Results written to model_results.json for H07_compare_models.py.
"""

import os
import cv2
import json
import numpy as np
import mediapipe as mp
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tqdm import tqdm

# ── Load splits ───────────────────────────────────────────────────────────────
with open('splits.pkl', 'rb') as f:
    X_train_paths, X_val_paths, X_test_paths, y_train, y_val, y_test = pickle.load(f)

all_paths  = X_train_paths + X_val_paths + X_test_paths
all_labels = y_train + y_val + y_test

# ── Landmark extraction with wrist-relative normalisation ─────────────────────
if os.path.exists('landmark_cache.pkl'):
    print("Loading cached landmarks...")
    with open('landmark_cache.pkl', 'rb') as f:
        landmark_data, valid_labels, saved_classes = pickle.load(f)
    le = LabelEncoder()
    le.classes_ = saved_classes
    print(f"Loaded {len(landmark_data)} samples from cache.")
else:
    print("Extracting landmarks (this may take a while)...")
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1,
                            min_detection_confidence=0.5)

    def extract_landmarks(image_path):
        """
        Extract 63 wrist-relative normalised coordinates.
        Translate wrist (landmark 0) to origin, scale to [-1, 1].
        Ref: Hussain et al. (2022, CMC) Pages 3–4, Section 3.2.
        """
        image = cv2.imread(image_path)
        if image is None:
            return None
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results   = hands.process(image_rgb)
        if results.multi_hand_landmarks:
            lm     = results.multi_hand_landmarks[0]
            coords = np.array([[p.x, p.y, p.z] for p in lm.landmark], dtype=np.float32)
            coords -= coords[0]                        # wrist to origin
            scale   = np.max(np.abs(coords)) + 1e-8
            coords /= scale                            # scale to [-1, 1]
            return coords.flatten()                    # 63 values
        return None

    landmark_data, valid_labels = [], []
    for path, label in tqdm(zip(all_paths, all_labels), total=len(all_paths),
                             desc="Extracting landmarks"):
        lm = extract_landmarks(path)
        if lm is not None:
            landmark_data.append(lm)
            valid_labels.append(label)

    hands.close()
    print(f"Extracted {len(landmark_data)} / {len(all_paths)} images")

    le = LabelEncoder()
    le.fit(valid_labels)
    with open('landmark_cache.pkl', 'wb') as f:
        pickle.dump((landmark_data, valid_labels, le.classes_), f)
    print("Cache saved → landmark_cache.pkl")

# ── Encode and split ──────────────────────────────────────────────────────────
y_encoded = le.transform(valid_labels)

X_lm_train, X_lm_temp, y_lm_train, y_lm_temp = train_test_split(
    landmark_data, y_encoded, test_size=0.30, stratify=y_encoded, random_state=42
)
X_lm_val, X_lm_test, y_lm_val, y_lm_test = train_test_split(
    X_lm_temp, y_lm_temp, test_size=0.50, random_state=42
)

num_classes = len(le.classes_)
print(f"Train: {len(X_lm_train)} | Val: {len(X_lm_val)} | Test: {len(X_lm_test)} | Classes: {num_classes}")

y_train_cat = to_categorical(y_lm_train, num_classes)
y_val_cat   = to_categorical(y_lm_val,   num_classes)

# ── Build MLP ─────────────────────────────────────────────────────────────────
model_lm = Sequential([
    Dense(256, activation='relu', input_shape=(63,)),
    BatchNormalization(), Dropout(0.3),
    Dense(128, activation='relu'),
    BatchNormalization(), Dropout(0.3),
    Dense(64,  activation='relu'),
    Dropout(0.2),
    Dense(num_classes, activation='softmax'),
], name='LandmarkMLP')

model_lm.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model_lm.summary()

callbacks = [
    EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1),
]

# ── Train ─────────────────────────────────────────────────────────────────────
history = model_lm.fit(
    np.array(X_lm_train), y_train_cat,
    validation_data=(np.array(X_lm_val), y_val_cat),
    epochs=60, batch_size=64, callbacks=callbacks, verbose=1,
)

# ── Evaluate ──────────────────────────────────────────────────────────────────
test_loss, test_acc = model_lm.evaluate(
    np.array(X_lm_test), to_categorical(y_lm_test, num_classes), verbose=0
)
print(f"\nLandmark MLP test accuracy : {test_acc:.4f}")
print(f"Landmark MLP test loss     : {test_loss:.4f}")

# ── Classification report ────────────────────────────────────────────────────
y_pred = np.argmax(model_lm.predict(np.array(X_lm_test), verbose=0), axis=1)
report = classification_report(y_lm_test, y_pred, target_names=le.classes_, digits=4)
print("\nClassification Report:\n", report)
with open('landmark_mlp_classification_report.txt', 'w') as f:
    f.write(report)
print("Classification report saved → landmark_mlp_classification_report.txt")

# ── Confusion matrix ──────────────────────────────────────────────────────────
cm = confusion_matrix(y_lm_test, y_pred)
fig, ax = plt.subplots(figsize=(16, 14))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=le.classes_, yticklabels=le.classes_,
            cmap='Blues', linewidths=0.4, ax=ax)
ax.set_xlabel('Predicted'); ax.set_ylabel('True')
ax.set_title(f'Landmark MLP — Confusion Matrix (Test Acc: {test_acc:.4f})')
plt.tight_layout()
plt.savefig('landmark_mlp_confusion_matrix.png', dpi=150); plt.close()
print("Confusion matrix saved → landmark_mlp_confusion_matrix.png")

# ── Training curves ───────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(history.history['accuracy'],     label='Train', color='steelblue')
axes[0].plot(history.history['val_accuracy'], label='Val',   color='darkorange')
axes[0].set_title('Accuracy'); axes[0].legend(); axes[0].grid(alpha=0.3)
axes[1].plot(history.history['loss'],         label='Train', color='steelblue')
axes[1].plot(history.history['val_loss'],     label='Val',   color='darkorange')
axes[1].set_title('Loss'); axes[1].legend(); axes[1].grid(alpha=0.3)
fig.suptitle('Landmark MLP Training Curves'); plt.tight_layout()
plt.savefig('landmark_mlp_training_curves.png', dpi=150); plt.close()
print("Training curves saved → landmark_mlp_training_curves.png")

# ── Update results JSON ───────────────────────────────────────────────────────
results = {}
if os.path.exists('model_results.json'):
    with open('model_results.json') as f:
        results = json.load(f)
results['Landmark MLP'] = round(float(test_acc), 4)
with open('model_results.json', 'w') as f:
    json.dump(results, f, indent=2)
print("Result saved → model_results.json")

# ── Save model and encoder ────────────────────────────────────────────────────
model_lm.save('landmark_mlp.h5')
with open('landmark_label_encoder.pkl', 'wb') as f:
    pickle.dump(le, f)
print("Model saved → landmark_mlp.h5")
print("Encoder saved → landmark_label_encoder.pkl")
