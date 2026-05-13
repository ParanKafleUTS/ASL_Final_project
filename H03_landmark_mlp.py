import os
import cv2
import numpy as np
import mediapipe as mp
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tqdm import tqdm

# Load splits produced by 02_data_validation.py
with open('splits.pkl', 'rb') as f:
    X_train_paths, X_val_paths, X_test_paths, y_train, y_val, y_test = pickle.load(f)

all_paths = X_train_paths + X_val_paths + X_test_paths
all_labels = y_train + y_val + y_test

# ----------------------------------------------------------------------
# Cache file to avoid re-extracting landmarks
cache_file = 'landmark_cache.pkl'

if os.path.exists(cache_file):
    print("Loading cached landmarks...")
    with open(cache_file, 'rb') as f:
        landmark_data, valid_labels, saved_classes = pickle.load(f)
    # Recreate label encoder with the saved classes
    le = LabelEncoder()
    le.classes_ = saved_classes
    print(f"Loaded {len(landmark_data)} landmark samples from cache.")
else:
    print("Extracting landmarks (this may take a long time)...")
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=0.5,
    )

    def extract_landmarks(image_path):
        image = cv2.imread(image_path)
        if image is None:
            return None
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)
        if results.multi_hand_landmarks:
            landmarks = results.multi_hand_landmarks[0]
            coords = []
            for lm in landmarks.landmark:
                coords.extend([lm.x, lm.y, lm.z])
            return np.array(coords)
        return None

    landmark_data = []
    valid_labels = []
    for path, label in tqdm(zip(all_paths, all_labels), total=len(all_paths), desc="Extracting landmarks"):
        lm = extract_landmarks(path)
        if lm is not None:
            landmark_data.append(lm)
            valid_labels.append(label)

    hands.close()
    print(f"Extracted landmarks for {len(landmark_data)} / {len(all_paths)} images")

    # Encode labels
    le = LabelEncoder()
    y_lm_encoded = le.fit_transform(valid_labels)   # not used directly, we keep valid_labels
    # We'll store the encoder classes and the raw labels (valid_labels) for later splitting
    # For caching we need the original valid_labels and the fitted classes
    with open(cache_file, 'wb') as f:
        pickle.dump((landmark_data, valid_labels, le.classes_), f)
    print(f"Saved landmarks to cache file: {cache_file}")

# Now we have landmark_data and valid_labels (strings) and le fitted
y_lm = le.transform(valid_labels)   # encode labels for splitting

# ----------------------------------------------------------------------
# Split: first split off 30% as temp (val+test)
X_lm_train, X_lm_temp, y_lm_train, y_lm_temp = train_test_split(
    landmark_data, y_lm, test_size=0.3, stratify=y_lm, random_state=42
)

# Second split: split temp into val and test (50% each) WITHOUT stratify
# (because some classes may have only 1 sample in temp)
X_lm_val, X_lm_test, y_lm_val, y_lm_test = train_test_split(
    X_lm_temp, y_lm_temp, test_size=0.5, random_state=42
)

num_classes = len(le.classes_)
y_lm_train_cat = to_categorical(y_lm_train, num_classes)
y_lm_val_cat = to_categorical(y_lm_val, num_classes)

# ----------------------------------------------------------------------
# Build the MLP
model_lm = Sequential([
    Dense(128, activation='relu', input_shape=(63,)),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(num_classes, activation='softmax'),
])
model_lm.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy'],
)

# Callbacks
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=4,
    restore_best_weights=True,
)
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=2,
    min_lr=1e-6,
)

# Train
history_lm = model_lm.fit(
    np.array(X_lm_train), y_lm_train_cat,
    validation_data=(np.array(X_lm_val), y_lm_val_cat),
    epochs=30,
    batch_size=32,
    callbacks=[early_stopping, reduce_lr],
)

# Evaluate
test_loss, test_acc = model_lm.evaluate(
    np.array(X_lm_test),
    to_categorical(y_lm_test, num_classes),
)
print(f"Landmark MLP test accuracy: {test_acc:.4f}")

# Save model and label encoder
model_lm.save('landmark_mlp.h5')
with open('landmark_label_encoder.pkl', 'wb') as f:
    pickle.dump(le, f)

print("Landmark model saved to landmark_mlp.h5")
print("Label encoder saved to landmark_label_encoder.pkl")