import os
import cv2
import mediapipe as mp
from tqdm import tqdm
import pickle
from sklearn.model_selection import train_test_split

# NOTE: Running MediaPipe on the full 87,000-image dataset is time-consuming
# (can take several hours on CPU).  If fast training is the priority, use
# 04_raw_cnn.py with data augmentation instead.  This script is provided for
# experiments that benefit from hand-region cropping.

IMG_SIZE = 224

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.5,
)

# Load original splits produced by 02_data_validation.py
with open('splits.pkl', 'rb') as f:
    X_train_paths, X_val_paths, X_test_paths, y_train, y_val, y_test = pickle.load(f)

all_paths  = X_train_paths + X_val_paths + X_test_paths
all_labels = y_train + y_val + y_test


def process_image(image_path):
    """Crop to hand region (with margin) using MediaPipe, then resize."""
    image = cv2.imread(image_path)
    if image is None:
        return None

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)
    h, w, _ = image.shape

    if results.multi_hand_landmarks:
        landmarks = results.multi_hand_landmarks[0]
        xs = [lm.x * w for lm in landmarks.landmark]
        ys = [lm.y * h for lm in landmarks.landmark]

        x_min = max(0, int(min(xs)) - 20)
        y_min = max(0, int(min(ys)) - 20)
        x_max = min(w, int(max(xs)) + 20)
        y_max = min(h, int(max(ys)) + 20)

        cropped = image[y_min:y_max, x_min:x_max]
        if cropped.size != 0:
            image = cropped

    return cv2.resize(image, (IMG_SIZE, IMG_SIZE))


cropped_dir = 'processed_images'
os.makedirs(cropped_dir, exist_ok=True)

processed_paths  = []
processed_labels = []

for path, label in tqdm(zip(all_paths, all_labels), total=len(all_paths), desc="Cropping"):
    processed = process_image(path)
    if processed is not None:
        class_dir = os.path.join(cropped_dir, label)
        os.makedirs(class_dir, exist_ok=True)
        new_path = os.path.join(class_dir, os.path.basename(path))
        cv2.imwrite(new_path, processed)
        processed_paths.append(new_path)
        processed_labels.append(label)

hands.close()
print(f"Total images saved: {len(processed_paths)}")

# New train / val / test split on the cropped images
X_train_new, X_temp, y_train_new, y_temp = train_test_split(
    processed_paths, processed_labels, test_size=0.3,
    stratify=processed_labels, random_state=42,
)
X_val_new, X_test_new, y_val_new, y_test_new = train_test_split(
    X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42,
)

with open('processed_splits.pkl', 'wb') as f:
    pickle.dump((X_train_new, X_val_new, X_test_new, y_train_new, y_val_new, y_test_new), f)

print("Processed splits saved to processed_splits.pkl")