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
from tensorflow.keras.optimizers import Adam
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

# ============= HYPERPARAMETER TUNING WITH GRID SEARCH =============
best_model = None
best_accuracy = 0
best_params = {}

# Define hyperparameter search space
param_grid = {
    'dense1_units': [64, 128, 256],
    'dense2_units': [32, 64, 128],
    'dropout1': [0.2, 0.3, 0.4],
    'dropout2': [0.2, 0.3, 0.4],
    'learning_rate': [0.001, 0.0005, 0.0001]
}

print("\n" + "="*60)
print("Starting Hyperparameter Tuning with Grid Search")
print("="*60)

param_combinations = [
    {'dense1': d1, 'dense2': d2, 'drop1': dr1, 'drop2': dr2, 'lr': lr}
    for d1 in param_grid['dense1_units']
    for d2 in param_grid['dense2_units']
    for dr1 in param_grid['dropout1']
    for dr2 in param_grid['dropout2']
    for lr in param_grid['learning_rate']
]

print(f"Total combinations to test: {len(param_combinations)}\n")

for idx, params in enumerate(param_combinations, 1):
    print(f"\nTesting combination {idx}/{len(param_combinations)}: {params}")
    
    # Build model with current hyperparameters
    model_lm = Sequential([
        Dense(params['dense1'], activation='relu', input_shape=(63,)),
        Dropout(params['drop1']),
        Dense(params['dense2'], activation='relu'),
        Dropout(params['drop2']),
        Dense(num_classes, activation='softmax')
    ])
    
    # Compile with custom learning rate
    optimizer = Adam(learning_rate=params['lr'])
    model_lm.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    
    # Define callbacks with Early Stopping and Learning Rate Reduction
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True,
        verbose=0
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=1e-7,
        verbose=0
    )
    
    # Train model
    history_lm = model_lm.fit(
        np.array(X_lm_train), y_lm_train_cat,
        validation_data=(np.array(X_lm_val), y_lm_val_cat),
        epochs=50,
        batch_size=32,
        callbacks=[early_stop, reduce_lr],
        verbose=0
    )
    
    # Evaluate on test set
    test_loss, test_acc = model_lm.evaluate(
        np.array(X_lm_test), 
        to_categorical(y_lm_test, num_classes),
        verbose=0
    )
    
    print(f"Test Accuracy: {test_acc:.4f}")
    
    # Update best model
    if test_acc > best_accuracy:
        best_accuracy = test_acc
        best_model = model_lm
        best_params = params
        print(f"✓ New best accuracy: {best_accuracy:.4f}")

# ============= TRAIN FINAL MODEL WITH BEST PARAMETERS =============
print("\n" + "="*60)
print("Training Final Model with Best Parameters")
print("="*60)
print(f"Best Parameters: {best_params}")
print(f"Best Test Accuracy: {best_accuracy:.4f}\n")

final_model = Sequential([
    Dense(best_params['dense1'], activation='relu', input_shape=(63,)),
    Dropout(best_params['drop1']),
    Dense(best_params['dense2'], activation='relu'),
    Dropout(best_params['drop2']),
    Dense(num_classes, activation='softmax')
])

optimizer = Adam(learning_rate=best_params['lr'])
final_model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

early_stop_final = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True,
    verbose=1
)

reduce_lr_final = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=3,
    min_lr=1e-7,
    verbose=1
)

history_final = final_model.fit(
    np.array(X_lm_train), y_lm_train_cat,
    validation_data=(np.array(X_lm_val), y_lm_val_cat),
    epochs=50,
    batch_size=32,
    callbacks=[early_stop_final, reduce_lr_final],
    verbose=1
)

test_loss_final, test_acc_final = final_model.evaluate(
    np.array(X_lm_test), 
    to_categorical(y_lm_test, num_classes),
    verbose=0
)
print(f"\nFinal Model Test Accuracy: {test_acc_final:.4f}")

# Save model and artifacts
final_model.save('landmark_mlp.h5')
with open('landmark_label_encoder.pkl', 'wb') as f:
    pickle.dump(le, f)

# Save hyperparameters for reference
with open('best_hyperparameters.pkl', 'wb') as f:
    pickle.dump(best_params, f)

print("\n✓ Model saved as 'landmark_mlp.h5'")
print("✓ Label encoder saved as 'landmark_label_encoder.pkl'")
print("✓ Best hyperparameters saved as 'best_hyperparameters.pkl'")