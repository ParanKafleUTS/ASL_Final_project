import cv2
import numpy as np
import mediapipe as mp
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tqdm import tqdm

# Load splits
with open('splits.pkl', 'rb') as f:
    X_train_paths, X_val_paths, X_test_paths, y_train, y_val, y_test = pickle.load(f)

all_paths = X_train_paths + X_val_paths + X_test_paths
all_labels = y_train + y_val + y_test

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)

def extract_landmarks(image_path):
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)
    if results.multi_hand_landmarks:
        landmarks = results.multi_hand_landmarks[0]
        coords = []
        for lm in landmarks.landmark:
            coords.extend([lm.x, lm.y, lm.z])
        return np.array(coords)
    else:
        return None

landmark_data = []
valid_labels = []
for path, label in tqdm(zip(all_paths, all_labels), total=len(all_paths)):
    lm = extract_landmarks(path)
    if lm is not None:
        landmark_data.append(lm)
        valid_labels.append(label)

print(f"Extracted landmarks for {len(landmark_data)} images")

le = LabelEncoder()
y_lm = le.fit_transform(valid_labels)

X_lm_train, X_lm_temp, y_lm_train, y_lm_temp = train_test_split(
    landmark_data, y_lm, test_size=0.3, stratify=y_lm, random_state=42
)
X_lm_val, X_lm_test, y_lm_val, y_lm_test = train_test_split(
    X_lm_temp, y_lm_temp, test_size=0.5, stratify=y_lm_temp, random_state=42
)

num_classes = len(le.classes_)
y_lm_train_cat = to_categorical(y_lm_train, num_classes)
y_lm_val_cat = to_categorical(y_lm_val, num_classes)

model_lm = Sequential([
    Dense(128, activation='relu', input_shape=(63,)),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(num_classes, activation='softmax')
])
model_lm.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

history_lm = model_lm.fit(
    np.array(X_lm_train), y_lm_train_cat,
    validation_data=(np.array(X_lm_val), y_lm_val_cat),
    epochs=30, batch_size=32
)

test_loss, test_acc = model_lm.evaluate(np.array(X_lm_test), to_categorical(y_lm_test, num_classes))
print(f"Landmark model test accuracy: {test_acc:.4f}")

# Save model
model_lm.save('landmark_mlp.h5')
with open('landmark_label_encoder.pkl', 'wb') as f:
    pickle.dump(le, f)