"""
H11_tsne.py
────────────
t-SNE visualisation of learned CNN feature embeddings.

Reads  : mobilenetv2_crop.h5  (or any CNN .h5)
         processed_splits.pkl
Writes : tsne_features.png     (2D scatter, coloured by class)
         tsne_features_3d.png  (3D scatter for appendix)
"""

import os
import pickle
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import tensorflow as tf
from sklearn.manifold import TSNE

MODEL_PATH  = 'mobilenetv2_crop.h5'
SPLITS_FILE = 'processed_splits.pkl'
IMG_SIZE    = 224
N_SAMPLES   = 600
RANDOM_SEED = 42
PERPLEXITY  = 35

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# ── Load model, strip classification head ─────────────────────────────────────
print(f"Loading: {MODEL_PATH}")
full_model = tf.keras.models.load_model(MODEL_PATH, compile=False)

gap_layer = next(
    (l.name for l in full_model.layers if 'global_average_pooling' in l.name.lower()),
    None
)
if gap_layer is None:
    raise ValueError("GlobalAveragePooling2D not found in model.")
print(f"Feature layer: {gap_layer}")

feature_extractor = tf.keras.Model(
    inputs=full_model.input,
    outputs=full_model.get_layer(gap_layer).output,
)

# ── Load test split ───────────────────────────────────────────────────────────
with open(SPLITS_FILE, 'rb') as f:
    _, _, X_test, _, _, y_test = pickle.load(f)

indices       = random.sample(range(len(X_test)), min(N_SAMPLES, len(X_test)))
sample_paths  = [X_test[i] for i in indices]
sample_labels = [y_test[i] for i in indices]
print(f"Embedding {len(sample_paths)} test images...")

# ── Preprocess ────────────────────────────────────────────────────────────────
model_lower = MODEL_PATH.lower()

def load_and_preprocess(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_image(img, channels=3, expand_animations=False)
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
    img = tf.cast(img, tf.float32)
    if 'mobilenetv2' in model_lower:
        img = (img / 127.5) - 1.0
    return img

batch    = np.array([load_and_preprocess(p).numpy() for p in sample_paths])
features = feature_extractor.predict(batch, batch_size=64, verbose=1)
print(f"Feature shape: {features.shape}")

# ── t-SNE 2D ─────────────────────────────────────────────────────────────────
print("Running t-SNE 2D...")
emb_2d = TSNE(n_components=2, perplexity=PERPLEXITY, learning_rate='auto',
              init='pca', random_state=RANDOM_SEED, max_iter=1500, verbose=1
              ).fit_transform(features)

unique_labels = sorted(set(sample_labels))
cmap          = plt.cm.get_cmap('tab20', len(unique_labels))

fig, ax = plt.subplots(figsize=(14, 11))
for i, lbl in enumerate(unique_labels):
    mask = np.array([l == lbl for l in sample_labels])
    ax.scatter(emb_2d[mask, 0], emb_2d[mask, 1],
               c=[cmap(i)], label=lbl, s=22, alpha=0.78, linewidths=0)
    cx, cy = np.mean(emb_2d[mask, 0]), np.mean(emb_2d[mask, 1])
    ax.text(cx, cy, lbl, fontsize=8, fontweight='bold', ha='center', va='center',
            bbox=dict(boxstyle='round,pad=0.15', fc='white', alpha=0.6, lw=0))
ax.set_title(
    f't-SNE of {features.shape[1]}-dim Features\n'
    f'({MODEL_PATH}, n={len(sample_paths)})', fontsize=13)
ax.set_xlabel('t-SNE Dim 1'); ax.set_ylabel('t-SNE Dim 2')
ax.legend(handles=[mpatches.Patch(color=cmap(i), label=lbl)
                   for i, lbl in enumerate(unique_labels)],
          bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=8, ncol=2)
plt.tight_layout()
plt.savefig('tsne_features.png', dpi=150, bbox_inches='tight'); plt.close()
print("t-SNE 2D saved → tsne_features.png")

# ── t-SNE 3D ─────────────────────────────────────────────────────────────────
print("Running t-SNE 3D...")
emb_3d = TSNE(n_components=3, perplexity=PERPLEXITY, learning_rate='auto',
              init='pca', random_state=RANDOM_SEED, max_iter=1500
              ).fit_transform(features)

fig = plt.figure(figsize=(13, 10))
ax  = fig.add_subplot(111, projection='3d')
for i, lbl in enumerate(unique_labels):
    mask = np.array([l == lbl for l in sample_labels])
    ax.scatter(emb_3d[mask, 0], emb_3d[mask, 1], emb_3d[mask, 2],
               c=[cmap(i)], label=lbl, s=18, alpha=0.75)
ax.set_title(f't-SNE 3D — {MODEL_PATH}', fontsize=12)
ax.set_xlabel('Dim 1'); ax.set_ylabel('Dim 2'); ax.set_zlabel('Dim 3')
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=7, ncol=2)
plt.tight_layout()
plt.savefig('tsne_features_3d.png', dpi=120, bbox_inches='tight'); plt.close()
print("t-SNE 3D saved → tsne_features_3d.png")
