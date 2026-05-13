"""
H10_latency_benchmark.py
─────────────────────────
Measures per-model inference latency (CPU, batch=1).

Reads  : *.h5 model files (current working directory)
Writes : latency_benchmark.json
         latency_benchmark.png
"""

import os
import json
import time
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

WARMUP_RUNS    = 15
BENCHMARK_RUNS = 300

# Models and their input shapes — all in the current working directory
MODELS = {
    'landmark_mlp.h5'           : (63,),
    'mobilenetv2_raw.h5'        : (224, 224, 3),
    'efficientnetb0_raw.h5'     : (224, 224, 3),
    'mobilenetv2_crop.h5'       : (224, 224, 3),
    'efficientnetb0_crop.h5'    : (224, 224, 3),
    'mobilenetv2_skeleton.h5'   : (224, 224, 3),
    'efficientnetb0_skeleton.h5': (224, 224, 3),
}


def benchmark_model(model_path, input_shape):
    if not os.path.exists(model_path):
        print(f"  [SKIP] {model_path} not found.")
        return None
    print(f"  Loading {model_path}...")
    model = tf.keras.models.load_model(model_path, compile=False)
    dummy = np.random.rand(1, *input_shape).astype(np.float32)

    for _ in range(WARMUP_RUNS):
        _ = model.predict(dummy, verbose=0)

    times_ms = []
    for _ in range(BENCHMARK_RUNS):
        t0 = time.perf_counter()
        _  = model.predict(dummy, verbose=0)
        times_ms.append((time.perf_counter() - t0) * 1000)

    times_ms = np.array(times_ms)
    result = {
        'mean_ms'  : round(float(np.mean(times_ms)),           2),
        'median_ms': round(float(np.median(times_ms)),         2),
        'std_ms'   : round(float(np.std(times_ms)),            2),
        'p95_ms'   : round(float(np.percentile(times_ms, 95)), 2),
        'max_fps'  : round(1000 / float(np.mean(times_ms)),    1),
    }
    print(f"    Mean: {result['mean_ms']:.2f} ms  "
          f"Median: {result['median_ms']:.2f} ms  "
          f"P95: {result['p95_ms']:.2f} ms  "
          f"Max FPS: {result['max_fps']:.1f}")
    del model; tf.keras.backend.clear_session()
    return result


print(f"Benchmarking {len(MODELS)} models ({BENCHMARK_RUNS} runs each, CPU)\n")
all_results = {}
for fname, shape in MODELS.items():
    res = benchmark_model(fname, shape)
    if res is not None:
        all_results[fname] = res

# Save JSON
with open('latency_benchmark.json', 'w') as f:
    json.dump(all_results, f, indent=2)
print(f"\nResults saved → latency_benchmark.json")

# Printed table
print("\n" + "─"*75)
print(f"{'Model':<38} {'Mean ms':>9} {'Median':>8} {'P95':>8} {'FPS':>7}")
print("─"*75)
for name, r in all_results.items():
    print(f"{name:<38} {r['mean_ms']:>9.2f} {r['median_ms']:>8.2f} "
          f"{r['p95_ms']:>8.2f} {r['max_fps']:>7.1f}")
print("─"*75)

# Chart
names       = list(all_results.keys())
means       = [all_results[n]['mean_ms'] for n in names]
stds        = [all_results[n]['std_ms']  for n in names]
max_fps     = [all_results[n]['max_fps'] for n in names]
short_names = [n.replace('.h5', '').replace('_', '\n') for n in names]

fig, axes = plt.subplots(1, 2, figsize=(16, 5))
axes[0].barh(short_names, means, xerr=stds, color='steelblue', edgecolor='white', capsize=4)
axes[0].set_xlabel('Mean Inference Time (ms)')
axes[0].set_title('Per-Model Latency (CPU, batch=1)')
axes[0].axvline(33.3, color='tomato', linestyle='--', linewidth=1, label='30 fps')
axes[0].legend(fontsize=9); axes[0].grid(axis='x', alpha=0.3)

colours = ['seagreen' if fps >= 30 else 'tomato' for fps in max_fps]
axes[1].barh(short_names, max_fps, color=colours, edgecolor='white')
axes[1].axvline(30, color='gray', linestyle='--', linewidth=1.2, label='30 fps threshold')
axes[1].set_xlabel('Estimated Max FPS')
axes[1].set_title('Throughput — Green ≥ 30 fps (real-time)')
axes[1].legend(fontsize=9); axes[1].grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig('latency_benchmark.png', dpi=150); plt.close()
print("Latency chart saved → latency_benchmark.png")
