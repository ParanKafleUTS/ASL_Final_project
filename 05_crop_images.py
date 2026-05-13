import os
import cv2
import mediapipe as mp
from tqdm import tqdm
import pickle
from sklearn.model_selection import train_test_split

IMG_SIZE = 224

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.5
)

# Load original splits
with open('splits.pkl', 'rb') as f:
    X_train_paths, X_val_paths, X_test_paths, y_train, y_val, y_test = pickle.load(f)

all_paths = X_train_paths + X_val_paths + X_test_paths
all_labels = y_train + y_val + y_test


def process_image(image_path):
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

        x_min, x_max = int(min(xs)), int(max(xs))
        y_min, y_max = int(min(ys)), int(max(ys))

        margin = 20
        x_min = max(0, x_min - margin)
        y_min = max(0, y_min - margin)
        x_max = min(w, x_max + margin)
        y_max = min(h, y_max + margin)

        cropped = image[y_min:y_max, x_min:x_max]

        if cropped.size != 0:
            image = cropped

    # Resize regardless (cropped or original)
    image_resized = cv2.resize(image, (IMG_SIZE, IMG_SIZE))

    return image_resized


cropped_dir = "processed_images"
os.makedirs(cropped_dir, exist_ok=True)

processed_paths = []
processed_labels = []

for path, label in tqdm(zip(all_paths, all_labels), total=len(all_paths)):

    processed = process_image(path)

    if processed is not None:

        class_dir = os.path.join(cropped_dir, label)
        os.makedirs(class_dir, exist_ok=True)

        new_filename = os.path.basename(path)
        new_path = os.path.join(class_dir, new_filename)

        cv2.imwrite(new_path, processed)

        processed_paths.append(new_path)
        processed_labels.append(label)

print(f"Total images saved: {len(processed_paths)}")


# Create new train/val/test splits
X_train_new, X_temp, y_train_new, y_temp = train_test_split(
    processed_paths,
    processed_labels,
    test_size=0.3,
    stratify=processed_labels,
    random_state=42
)

X_val_new, X_test_new, y_val_new, y_test_new = train_test_split(
    X_temp,
    y_temp,
    test_size=0.5,
    stratify=y_temp,
    random_state=42
)


with open("processed_splits.pkl", "wb") as f:
    pickle.dump(
        (
            X_train_new,
            X_val_new,
            X_test_new,
            y_train_new,
            y_val_new,
            y_test_new,
        ),
        f,
    )

print("New dataset splits saved.")