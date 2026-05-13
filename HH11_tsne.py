"""
H11_tsne.py
────────────
Runs t-SNE feature embedding visualisation for ALL trained models.

For each model, extracts the GlobalAveragePooling2D feature vector
from test images, reduces to 2D with t-SNE, and saves a scatter plot
coloured by class label.

Also produces a final side-by-side comparison grid of all models.

Reads  : splits.pkl              (raw models  -- from H02)
         processed_splits.pkl    (crop models -- from H05)
         skeleton_images/        (skeleton models -- from Hgenerate)
         *.h5 model files
Writes : tsne_<modelname>.png         (one 2D plot per model)
         tsne_<modelname>_3d.png      (one 3D plot per model)
         tsne_comparison_grid.png     (all 2D plots side by side)
"""

import os
import pickle
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import tensorflow as tf
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from collections import Counter

# ── Configuration ──────────────────────────────────────────────────────────────
IMG_SIZE    = 224
N_SAMPLES   = 500     # per model
RANDOM_SEED = 42
PERPLEXITY  = 30
MAX_ITER    = 1000

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# ── Model registry ─────────────────────────────────────────────────────────────
# (model filename, splits source, preprocessing type, short display label)
#
# Preprocessing types:
#   'mnv2'         -> (x / 127.5) - 1.0   MobileNetV2 needs [-1, 1]
#   'efficientnet' -> raw [0, 255]         EfficientNet handles internally
MODELS = [
    ('mobilenetv2_raw.h5',         'splits.pkl',           'mnv2',         'MobileNetV2 Raw'),
    ('efficientnetb0_raw.h5',      'splits.pkl',           'efficientnet', 'EfficientNetB0 Raw'),
    ('mobilenetv2_crop.h5',        'processed_splits.pkl', 'mnv2',         'MobileNetV2 Crop'),
    ('efficientnetb0_crop.h5',     'processed_splits.pkl', 'efficientnet', 'EfficientNetB0 Crop'),
    ('mobilenetv2_skeleton.h5',    'skeleton',             'mnv2',         'MobileNetV2 Skeleton'),
    ('efficientnetb0_skeleton.h5', 'skeleton',             'efficientnet', 'EfficientNetB0 Skeleton'),
]

SKELETON_DIR = 'skeleton_images'


# ── Helpers ────────────────────────────────────────────────────────────────────
def load_feature_extractor(model_path):
    """Load model and return a feature extractor at GlobalAveragePooling2D."""
    print(f"  Loading {model_path} ...")
    model = tf.keras.models.load_model(model_path, compile=False)
    gap_name = next(
        (l.name for l in model.layers if 'global_average_pooling' in l.name.lower()),
        None
    )
    if gap_name is None:
        raise ValueError(f"GlobalAveragePooling2D not found in {model_path}")
    feat_dim = model.get_layer(gap_name).output_shape[-1]
    print(f"  Feature layer: {gap_name}  |  dim: {feat_dim}")
    extractor = tf.keras.Model(inputs=model.input,
                               outputs=model.get_layer(gap_name).output)
    return extractor, feat_dim


def preprocess_image(path, preprocess_type):
    img = tf.io.read_file(path)
    img = tf.image.decode_image(img, channels=3, expand_animations=False)
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
    img = tf.cast(img, tf.float32)
    if preprocess_type == 'mnv2':
        img = (img / 127.5) - 1.0
    # efficientnet: leave as raw [0, 255]
    return img


def get_test_samples(splits_src, n_samples):
    """Return (paths, labels) for the test split of the given source."""
    if splits_src == 'skeleton':
        from imutils import paths as imp
        all_paths  = list(imp.list_images(SKELETON_DIR))
        all_labels = [os.path.basename(os.path.dirname(p)) for p in all_paths]
        counts     = Counter(all_labels)
        pairs      = [(p, l) for p, l in zip(all_paths, all_labels) if counts[l] >= 2]
        all_paths  = [x[0] for x in pairs]
        all_labels = [x[1] for x in pairs]
        _, temp_p, _, temp_l = train_test_split(
            all_paths, all_labels, test_size=0.30,
            stratify=all_labels, random_state=RANDOM_SEED)
        _, X_test, _, y_test = train_test_split(
            temp_p, temp_l, test_size=0.50, random_state=RANDOM_SEED)
    else:
        with open(splits_src, 'rb') as f:
            _, _, X_test, _, _, y_test = pickle.load(f)

    idx = random.sample(range(len(X_test)), min(n_samples, len(X_test)))
    return [X_test[i] for i in idx], [y_test[i] for i in idx]


def extract_features(extractor, paths, preprocess_type):
    batch = np.array([preprocess_image(p, preprocess_type).numpy() for p in paths])
    return extractor.predict(batch, batch_size=64, verbose=1)


def build_colours(unique_labels):
    n = len(unique_labels)
    if n <= 20:
        cmap = plt.cm.get_cmap('tab20', n)
        return [cmap(i) for i in range(n)]
    cmap1 = plt.cm.get_cmap('tab20',  20)
    cmap2 = plt.cm.get_cmap('tab20b', n - 20)
    return [cmap1(i) for i in range(20)] + [cmap2(i) for i in range(n - 20)]


def plot_tsne_2d(emb, labels, title, save_path, figsize=(13, 10)):
    unique_labels = sorted(set(labels))
    colours       = build_colours(unique_labels)

    fig, ax = plt.subplots(figsize=figsize)
    for i, lbl in enumerate(unique_labels):
        mask = np.array([l == lbl for l in labels])
        ax.scatter(emb[mask, 0], emb[mask, 1],
                   c=[colours[i]], s=22, alpha=0.82, linewidths=0)
        cx = np.mean(emb[mask, 0])
        cy = np.mean(emb[mask, 1])
        ax.text(cx, cy, lbl, fontsize=7.5, fontweight='bold',
                ha='center', va='center',
                bbox=dict(boxstyle='round,pad=0.13', fc='white', alpha=0.65, lw=0))

    ax.set_title(title, fontsize=12, pad=10)
    ax.set_xlabel('t-SNE Dim 1', fontsize=9)
    ax.set_ylabel('t-SNE Dim 2', fontsize=9)
    ax.tick_params(labelsize=7)
    handles = [mpatches.Patch(color=colours[i], label=lbl)
               for i, lbl in enumerate(unique_labels)]
    ax.legend(handles=handles, bbox_to_anchor=(1.01, 1),
              loc='upper left', fontsize=7, ncol=2)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  2D saved -> {save_path}")


def plot_tsne_3d(emb, labels, title, save_path):
    unique_labels = sorted(set(labels))
    colours       = build_colours(unique_labels)

    fig = plt.figure(figsize=(12, 9))
    ax  = fig.add_subplot(111, projection='3d')
    for i, lbl in enumerate(unique_labels):
        mask = np.array([l == lbl for l in labels])
        ax.scatter(emb[mask, 0], emb[mask, 1], emb[mask, 2],
                   c=[colours[i]], s=16, alpha=0.75)

    ax.set_title(title, fontsize=11)
    ax.set_xlabel('Dim 1', fontsize=8)
    ax.set_ylabel('Dim 2', fontsize=8)
    ax.set_zlabel('Dim 3', fontsize=8)
    handles = [mpatches.Patch(color=colours[i], label=lbl)
               for i, lbl in enumerate(unique_labels)]
    ax.legend(handles=handles, bbox_to_anchor=(1.05, 1),
              loc='upper left', fontsize=6, ncol=2)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"  3D saved -> {save_path}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN LOOP — run t-SNE for every model
# ══════════════════════════════════════════════════════════════════════════════
completed = []   # (display_label, emb_2d, labels, feat_dim) for grid
skipped   = []

for model_file, splits_src, preprocess_type, display_label in MODELS:

    print("\n" + "=" * 60)
    print(f"  {display_label}  ({model_file})")
    print("=" * 60)

    # Check prerequisites
    if not os.path.exists(model_file):
        print(f"  [SKIP] {model_file} not found.")
        skipped.append(display_label); continue

    if splits_src == 'skeleton' and not os.path.isdir(SKELETON_DIR):
        print(f"  [SKIP] {SKELETON_DIR}/ not found.")
        skipped.append(display_label); continue

    if splits_src != 'skeleton' and not os.path.exists(splits_src):
        print(f"  [SKIP] {splits_src} not found.")
        skipped.append(display_label); continue

    try:
        extractor, feat_dim = load_feature_extractor(model_file)
        paths, labels       = get_test_samples(splits_src, N_SAMPLES)
        print(f"  Samples: {len(paths)}  |  Classes: {len(set(labels))}")

        features = extract_features(extractor, paths, preprocess_type)
        print(f"  Feature shape: {features.shape}")

        tag = model_file.replace('.h5', '')

        # 2D t-SNE
        print("  Running t-SNE 2D ...")
        emb_2d   = TSNE(n_components=2, perplexity=PERPLEXITY,
                        learning_rate='auto', init='pca',
                        random_state=RANDOM_SEED, max_iter=MAX_ITER,
                        verbose=0).fit_transform(features)
        title_2d = (f"t-SNE — {display_label}  "
                    f"({feat_dim}-dim features, n={len(paths)})")
        plot_tsne_2d(emb_2d, labels, title_2d, f'tsne_{tag}.png')

        # 3D t-SNE
        print("  Running t-SNE 3D ...")
        emb_3d = TSNE(n_components=3, perplexity=PERPLEXITY,
                      learning_rate='auto', init='pca',
                      random_state=RANDOM_SEED, max_iter=MAX_ITER,
                      verbose=0).fit_transform(features)
        plot_tsne_3d(emb_3d, labels, f"t-SNE 3D -- {display_label}",
                     f'tsne_{tag}_3d.png')

        completed.append((display_label, emb_2d, labels, feat_dim))

        del extractor, features, emb_2d, emb_3d
        tf.keras.backend.clear_session()

    except Exception as e:
        print(f"  [ERROR] {e}")
        skipped.append(display_label)
        tf.keras.backend.clear_session()
        continue


# ══════════════════════════════════════════════════════════════════════════════
# COMPARISON GRID — all completed models in one figure
# ══════════════════════════════════════════════════════════════════════════════
if len(completed) >= 2:
    print("\n" + "=" * 60)
    print("  Building comparison grid ...")
    print("=" * 60)

    n   = len(completed)
    nc  = min(3, n)
    nr  = (n + nc - 1) // nc

    fig, axes = plt.subplots(nr, nc, figsize=(nc * 6.5, nr * 6))
    axes      = np.array(axes).reshape(-1)

    for idx, (display_label, emb_2d, labels, feat_dim) in enumerate(completed):
        ax            = axes[idx]
        unique_labels = sorted(set(labels))
        colours       = build_colours(unique_labels)

        for i, lbl in enumerate(unique_labels):
            mask = np.array([l == lbl for l in labels])
            ax.scatter(emb_2d[mask, 0], emb_2d[mask, 1],
                       c=[colours[i]], s=14, alpha=0.82, linewidths=0)
            cx = np.mean(emb_2d[mask, 0])
            cy = np.mean(emb_2d[mask, 1])
            ax.text(cx, cy, lbl, fontsize=5.5, fontweight='bold',
                    ha='center', va='center',
                    bbox=dict(boxstyle='round,pad=0.1', fc='white',
                              alpha=0.60, lw=0))

        ax.set_title(display_label, fontsize=11, fontweight='bold', pad=5)
        ax.set_xlabel('Dim 1', fontsize=7)
        ax.set_ylabel('Dim 2', fontsize=7)
        ax.tick_params(labelsize=6)

    for idx in range(len(completed), len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle(
        't-SNE Feature Embedding Comparison -- All Models\n'
        f'({N_SAMPLES} test images per model, GlobalAveragePooling2D features)',
        fontsize=13, fontweight='bold', y=1.01
    )
    plt.tight_layout()
    plt.savefig('tsne_comparison_grid.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Comparison grid saved -> tsne_comparison_grid.png")


# ── Summary ───────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  SUMMARY")
print("=" * 60)
print(f"  Completed : {len(completed)}")
for label, _, _, dim in completed:
    tag = label.lower().replace(' ', '_')
    print(f"    {label:<30}  tsne_{tag}.png  |  tsne_{tag}_3d.png")

if skipped:
    print(f"\n  Skipped : {len(skipped)}")
    for s in skipped:
        print(f"    {s}")

print("\nKey output: tsne_comparison_grid.png (all models side by side)")
