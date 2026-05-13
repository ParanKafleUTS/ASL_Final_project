"""
H09_gradcam.py
──────────────
Grad-CAM explainability — highlights which pixels drove each prediction.

Reads  : mobilenetv2_crop.h5  (or any CNN .h5)
         processed_splits.pkl
Writes : gradcam_grid.png
         gradcam/<class>/sample_N.jpg

Ref: Selvaraju et al. (2017) — Grad-CAM.
"""

import os
import cv2
import pickle
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from collections import defaultdict

MODEL_PATH  = 'mobilenetv2_crop.h5'   # change to any trained CNN model
SPLITS_FILE = 'processed_splits.pkl'
OUTPUT_DIR  = 'gradcam'
IMG_SIZE    = 224
N_SAMPLES   = 3
ALPHA       = 0.45

LAST_CONV = {'mobilenetv2': 'Conv_1', 'efficientnetb0': 'top_conv'}


def make_gradcam_heatmap(img_array, model, conv_layer_name):
    grad_model = tf.keras.models.Model(
        inputs =model.inputs,
        outputs=[model.get_layer(conv_layer_name).output, model.output],
    )
    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(img_array)
        pred_idx        = tf.argmax(preds[0])
        class_score     = preds[:, pred_idx]
    grads        = tape.gradient(class_score, conv_out)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    heatmap      = conv_out[0] @ pooled_grads[..., tf.newaxis]
    heatmap      = tf.maximum(tf.squeeze(heatmap), 0)
    heatmap     /= (tf.math.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy(), int(pred_idx), float(preds[0][pred_idx])


def overlay_heatmap(img_bgr, heatmap):
    h, w = img_bgr.shape[:2]
    hm   = cv2.resize(heatmap, (w, h))
    hm   = cv2.applyColorMap(np.uint8(255 * hm), cv2.COLORMAP_JET)
    out  = cv2.addWeighted(img_bgr, 1 - ALPHA, hm, ALPHA, 0)
    return cv2.cvtColor(out, cv2.COLOR_BGR2RGB)


print(f"Loading: {MODEL_PATH}")
model       = tf.keras.models.load_model(MODEL_PATH, compile=False)
model_lower = MODEL_PATH.lower()
CONV_LAYER  = (LAST_CONV['mobilenetv2']   if 'mobilenetv2'   in model_lower
          else LAST_CONV['efficientnetb0'] if 'efficientnet'  in model_lower
          else [l.name for l in model.layers if 'conv' in l.name.lower()][-1])
print(f"Grad-CAM layer: {CONV_LAYER}")

with open(SPLITS_FILE, 'rb') as f:
    _, _, X_test, _, _, y_test = pickle.load(f)

class_paths = defaultdict(list)
for path, label in zip(X_test, y_test):
    class_paths[label].append(path)

sorted_classes = sorted(class_paths.keys())
os.makedirs(OUTPUT_DIR, exist_ok=True)


def preprocess(img_path):
    img = cv2.imread(img_path)
    if img is None: return None, None
    img_r = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    rgb   = cv2.cvtColor(img_r, cv2.COLOR_BGR2RGB).astype(np.float32)
    inp   = ((rgb / 127.5) - 1.0) if 'mobilenetv2' in model_lower else rgb
    return np.expand_dims(inp, 0), img_r


fig_rows = len(sorted_classes)
fig_cols = N_SAMPLES * 2
fig = plt.figure(figsize=(fig_cols * 2.2, fig_rows * 2.4))
fig.suptitle(f'Grad-CAM — {MODEL_PATH}', fontsize=14, y=1.01)
plot_idx = 1

print(f"Generating Grad-CAM for {len(sorted_classes)} classes...")
for cls in sorted_classes:
    paths   = class_paths[cls][:N_SAMPLES]
    cls_dir = os.path.join(OUTPUT_DIR, cls)
    os.makedirs(cls_dir, exist_ok=True)

    for i, img_path in enumerate(paths):
        inp, img_bgr = preprocess(img_path)
        if inp is None: continue
        heatmap, _, conf = make_gradcam_heatmap(inp, model, CONV_LAYER)
        overlay = overlay_heatmap(img_bgr, heatmap)
        cv2.imwrite(os.path.join(cls_dir, f'sample_{i}.jpg'),
                    cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

        ax1 = fig.add_subplot(fig_rows, fig_cols, plot_idx)
        ax1.imshow(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
        ax1.set_title(f'{cls}', fontsize=7, fontweight='bold'); ax1.axis('off')
        plot_idx += 1
        ax2 = fig.add_subplot(fig_rows, fig_cols, plot_idx)
        ax2.imshow(overlay)
        ax2.set_title(f'{conf:.0%}', fontsize=7); ax2.axis('off')
        plot_idx += 1

plt.tight_layout()
plt.savefig('gradcam_grid.png', dpi=120, bbox_inches='tight'); plt.close()
print("Grad-CAM grid saved → gradcam_grid.png")
print(f"Individual overlays → {OUTPUT_DIR}/")
