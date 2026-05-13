import matplotlib.pyplot as plt
import pickle
import numpy as np

# Collect accuracies (you can also load them from training logs)
# For simplicity, we manually list them here (replace with actual values from your runs)
accuracies = {
    'Landmark MLP': 0.86,      # replace with actual test_acc from step 4
    'MobileNetV2 Raw': 0.89,   # replace
    'EfficientNetB0 Raw': 0.03,
    'MobileNetV2 Crop': 0.91,
    'EfficientNetB0 Crop': 0.04
}

models = list(accuracies.keys())
scores = list(accuracies.values())

plt.figure(figsize=(10,6))
bars = plt.bar(models, scores, color=['blue','green','green','orange','orange'])
plt.ylabel('Test Accuracy')
plt.title('Model Comparison')
for bar, acc in zip(bars, scores):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f'{acc:.3f}', ha='center')
plt.ylim(0, 1.1)
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig('model_comparison.png')
plt.show()

# Identify best model
best_model_name = max(accuracies, key=accuracies.get)
print(f"Best model: {best_model_name} with accuracy {accuracies[best_model_name]:.4f}")