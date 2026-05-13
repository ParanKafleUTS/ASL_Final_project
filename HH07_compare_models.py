"""
H07_compare_models.py
----------------------
Step 7 -- Load test accuracies from model_results.json and produce:
  - model_comparison.png   (colour-coded bar chart)
  - ablation_study.txt     (ablation study table)

Reads  : model_results.json  (written by H03, H04, H06, Hgenerate, HH04)
Writes : model_comparison.png
         ablation_study.txt
"""

import json
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

RESULTS_FILE = 'model_results.json'

# -- Load results --------------------------------------------------------------
if not os.path.exists(RESULTS_FILE):
    print(f"[WARNING] {RESULTS_FILE} not found.")
    print("Run H03, H04, H06, HH04, and Hgenerate first to populate results.")
    # Placeholder values -- replace with your actual results
    accuracies = {
        'Landmark MLP'           : 0.000,
        'MobileNetV2 Raw'        : 0.000,
        'EfficientNetB0 Raw'     : 0.000,
        'MobileNetV2 Crop'       : 0.000,
        'EfficientNetB0 Crop'    : 0.000,
        'MobileNetV2 Skeleton'   : 0.000,
        'EfficientNetB0 Skeleton': 0.000,
    }
else:
    with open(RESULTS_FILE) as f:
        accuracies = json.load(f)
    print(f"Loaded results from {RESULTS_FILE}")
    for k, v in sorted(accuracies.items(), key=lambda x: -x[1]):
        print(f"  {k:<30} : {v:.4f}")

# Preferred display order
ORDER = [
    'Landmark MLP',
    'MobileNetV2 Raw', 'EfficientNetB0 Raw',
    'MobileNetV2 Crop', 'EfficientNetB0 Crop',
    'MobileNetV2 Skeleton', 'EfficientNetB0 Skeleton',
]
models  = [m for m in ORDER if m in accuracies]
models += [m for m in accuracies if m not in models]
scores  = [accuracies[m] for m in models]

colour_map = {
    'Landmark MLP'           : '#4e79a7',
    'MobileNetV2 Raw'        : '#59a14f',
    'EfficientNetB0 Raw'     : '#76b7b2',
    'MobileNetV2 Crop'       : '#f28e2b',
    'EfficientNetB0 Crop'    : '#e15759',
    'MobileNetV2 Skeleton'   : '#b07aa1',
    'EfficientNetB0 Skeleton': '#ff9da7',
}
colours    = [colour_map.get(m, '#aec7e8') for m in models]
best_model = max(accuracies, key=accuracies.get)
best_score = accuracies[best_model]
best_idx   = models.index(best_model)

# -- Bar chart -----------------------------------------------------------------
fig, ax = plt.subplots(figsize=(14, 6))
bars = ax.bar(models, scores, color=colours, edgecolor='white', linewidth=0.8)
for bar, acc in zip(bars, scores):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.008,
            f'{acc:.4f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
bars[best_idx].set_edgecolor('gold'); bars[best_idx].set_linewidth(2.5)
ax.set_ylabel('Test Accuracy', fontsize=11)
ax.set_title('Model Comparison -- ASL Recognition', fontsize=13, fontweight='bold')
ax.set_ylim(0, 1.12)
ax.set_xticks(range(len(models)))
ax.set_xticklabels(models, rotation=20, ha='right', fontsize=9)
ax.axhline(best_score, color='gold', linestyle='--', linewidth=1,
           label=f'Best: {best_model} ({best_score:.4f})')
ax.grid(axis='y', alpha=0.3)
legend_patches = [
    mpatches.Patch(color='#4e79a7', label='Landmark-based'),
    mpatches.Patch(color='#59a14f', label='Raw CNN'),
    mpatches.Patch(color='#f28e2b', label='Cropped CNN'),
    mpatches.Patch(color='#b07aa1', label='Skeleton CNN'),
]
ax.legend(handles=legend_patches, loc='upper left', fontsize=9)
plt.tight_layout()
plt.savefig('model_comparison.png', dpi=150); plt.close()
print("\nComparison chart saved -> model_comparison.png")

# -- Ablation study table ------------------------------------------------------
def acc(key):
    return accuracies.get(key, float('nan'))

ablation_rows = [
    ('Full system (MobileNetV2 Crop + Fine-tune)', '-- (full system)',        acc('MobileNetV2 Crop')),
    ('No fine-tuning (Phase 1 only)',              'Remove fine-tune phase', acc('MobileNetV2 Raw')),
    ('No cropping (MobileNetV2 Raw)',              'Remove H05 cropping',    acc('MobileNetV2 Raw')),
    ('Landmark MLP (normalised)',                  'Replace CNN with MLP',   acc('Landmark MLP')),
    ('EfficientNetB0 Crop',                        'Swap backbone',          acc('EfficientNetB0 Crop')),
    ('EfficientNetB0 Raw',                         'Swap backbone + no crop',acc('EfficientNetB0 Raw')),
    ('MobileNetV2 Skeleton',                       'Use skeleton rendering', acc('MobileNetV2 Skeleton')),
]

header = f"{'Configuration':<50} {'Change':<28} {'Acc':>8}"
sep    = "-" * 90
lines  = [f"ABLATION STUDY -- ASL Recognition\n{'='*90}\n", header + "\n", sep + "\n"]

print("\n" + sep)
print(header)
print(sep)
for config, change, a in ablation_rows:
    s = f'{a:.4f}' if not np.isnan(a) else '  N/A'
    row = f"{config:<50} {change:<28} {s:>8}"
    print(row)
    lines.append(row + "\n")
print(sep)
lines += [sep + "\n", f"\nBest model: {best_model}  |  Accuracy: {best_score:.4f}\n"]

with open('ablation_study.txt', 'w', encoding='utf-8') as f:
    f.writelines(lines)
print("Ablation study saved -> ablation_study.txt")
print(f"\nBest model : {best_model}  |  Accuracy : {best_score:.4f}")
