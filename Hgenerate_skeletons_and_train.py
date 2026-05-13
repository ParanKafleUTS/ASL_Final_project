import os
import cv2
import numpy as np
import mediapipe as mp
from tqdm import tqdm
import pickle
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2, EfficientNetB0
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, Rescaling
from tensorflow.keras.layers import RandomRotation, RandomTranslation, RandomZoom, RandomBrightness
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# ------------------------------
# Configuration
# ------------------------------
if os.path.exists('train_folder.txt'):
    with open('train_folder.txt', 'r') as _f:
        SOURCE_DIR = _f.read().strip()
else:
    SOURCE_DIR = 'subset_5000'

SKELETON_DIR = 'skeleton_images'
IMG_SIZE = 224
BATCH_SIZE = 64  # Increased for faster tf.data processing
EPOCHS = 30
TEST_SIZE = 0.15
VAL_SIZE = 0.15
RANDOM_STATE = 42

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)

# ------------------------------
# Step 1: Generate skeleton images
# ------------------------------
def generate_skeleton_images():
    os.makedirs(SKELETON_DIR, exist_ok=True)
    classes = [d for d in os.listdir(SOURCE_DIR) if os.path.isdir(os.path.join(SOURCE_DIR, d))]
    total = 0
    skipped = 0

    for cls in classes:
        src_class_path = os.path.join(SOURCE_DIR, cls)
        dst_class_path = os.path.join(SKELETON_DIR, cls)
        os.makedirs(dst_class_path, exist_ok=True)

        images = [f for f in os.listdir(src_class_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        for img_file in tqdm(images, desc=f"Processing {cls}"):
            src_path = os.path.join(src_class_path, img_file)
            dst_path = os.path.join(dst_class_path, img_file)

            if os.path.exists(dst_path):
                total += 1
                continue

            image = cv2.imread(src_path)
            if image is None:
                skipped += 1
                continue

            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = hands.process(image_rgb)

            if results.multi_hand_landmarks:
                skeleton = np.ones((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8) * 255
                for hand_landmarks in results.multi_hand_landmarks:
                    points = [(int(lm.x * IMG_SIZE), int(lm.y * IMG_SIZE)) for lm in hand_landmarks.landmark]
                    for connection in mp_hands.HAND_CONNECTIONS:
                        cv2.line(skeleton, points[connection[0]], points[connection[1]], (0, 0, 0), 2)
                    for (x, y) in points:
                        cv2.circle(skeleton, (x, y), 3, (0, 0, 255), -1)

                cv2.imwrite(dst_path, skeleton)
                total += 1
            else:
                skipped += 1

    hands.close()
    print(f"Skeleton generation complete. Saved: {total}, Skipped (no hand): {skipped}")

# ------------------------------
# Step 2: Prepare data splits
# ------------------------------
# def prepare_splits():
#     image_paths = []
#     labels = []
#     classes = sorted([d for d in os.listdir(SKELETON_DIR) if os.path.isdir(os.path.join(SKELETON_DIR, d))])
    
#     for cls in classes:
#         class_path = os.path.join(SKELETON_DIR, cls)
#         images = [os.path.join(class_path, f) for f in os.listdir(class_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
#         image_paths.extend(images)
#         labels.extend([cls] * len(images))

#     X_train, X_temp, y_train, y_temp = train_test_split(
#         image_paths, labels, test_size=(VAL_SIZE + TEST_SIZE), stratify=labels, random_state=RANDOM_STATE
#     )
#     val_ratio = VAL_SIZE / (VAL_SIZE + TEST_SIZE)
#     X_val, X_test, y_val, y_test = train_test_split(
#         X_temp, y_temp, test_size=(1 - val_ratio), stratify=y_temp, random_state=RANDOM_STATE
#     )

#     return X_train, X_val, X_test, y_train, y_val, y_test, classes
# ------------------------------
# Step 2: Prepare data splits
# ------------------------------
def prepare_splits():
    image_paths = []
    labels = []
    all_classes = sorted([d for d in os.listdir(SKELETON_DIR) if os.path.isdir(os.path.join(SKELETON_DIR, d))])
    valid_classes = []
    
    for cls in all_classes:
        class_path = os.path.join(SKELETON_DIR, cls)
        images = [os.path.join(class_path, f) for f in os.listdir(class_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        # FIX: Skip classes that have too few images to safely stratify
        if len(images) < 10:
            print(f"Skipping class '{cls}' - only {len(images)} valid skeleton images found.")
            continue
            
        image_paths.extend(images)
        labels.extend([cls] * len(images))
        valid_classes.append(cls)

    print(f"Total valid images: {len(image_paths)} across {len(valid_classes)} classes.")

    X_train, X_temp, y_train, y_temp = train_test_split(
        image_paths, labels, test_size=(VAL_SIZE + TEST_SIZE), stratify=labels, random_state=RANDOM_STATE
    )
    val_ratio = VAL_SIZE / (VAL_SIZE + TEST_SIZE)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=(1 - val_ratio), stratify=y_temp, random_state=RANDOM_STATE
    )

    return X_train, X_val, X_test, y_train, y_val, y_test, valid_classes

# ------------------------------
# Step 3: Fast tf.data Pipeline & Model
# ------------------------------
AUTOTUNE = tf.data.AUTOTUNE

def process_path(file_path, label):
    img = tf.io.read_file(file_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
    return img, label

# def create_dataset(paths, labels, is_training=False):
#     ds = tf.data.Dataset.from_tensor_slices((paths, labels))
#     if is_training:
#         ds = ds.shuffle(buffer_size=10000)
#     ds = ds.map(process_path, num_parallel_calls=AUTOTUNE)
#     ds = ds.batch(BATCH_SIZE)
#     ds = ds.prefetch(buffer_size=AUTOTUNE)
#     return ds
def create_dataset(paths, labels, is_training=False):
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    ds = ds.map(process_path, num_parallel_calls=AUTOTUNE)
    
    # -----------------------------------------------------------
    # THIS LINE SAVES MASSIVE TIME ON EPOCHS 2-30
    # It stores the resized images in RAM so it doesn't read the disk again
    ds = ds.cache() 
    # -----------------------------------------------------------
    
    if is_training:
        ds = ds.shuffle(buffer_size=10000)
        
    ds = ds.batch(BATCH_SIZE)
    ds = ds.prefetch(buffer_size=AUTOTUNE)
    return ds

data_augmentation = Sequential([
    RandomRotation(0.04),
    RandomTranslation(0.1, 0.1),
    RandomZoom(0.1),
    RandomBrightness(0.2)
], name="data_augmentation")

def create_fast_model(base_model_cls, num_classes, model_type):
    inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = data_augmentation(inputs)
    
    # ---------------------------------------------------------
    # THIS FIXES THE PROBLEM: Handle scaling dynamically!
    # ---------------------------------------------------------
    if model_type == 'mobilenetv2':
        x = Rescaling(scale=1./127.5, offset=-1)(x) # [-1, 1] for MobileNet
    # EfficientNet gets raw [0, 255] directly, no Rescaling needed
    
    base = base_model_cls(weights='imagenet', include_top=False, input_tensor=x)
    base.trainable = False
    
    x = base.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.3)(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs, outputs)
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy', # 'sparse' because labels are integers
        metrics=['accuracy'],
    )
    return model

# ------------------------------
# Main execution
# ------------------------------
if __name__ == "__main__":
    skeleton_has_classes = os.path.exists(SKELETON_DIR) and any(os.path.isdir(os.path.join(SKELETON_DIR, d)) for d in os.listdir(SKELETON_DIR))
    if not skeleton_has_classes:
        print("Generating skeleton images...")
        generate_skeleton_images()
    else:
        print("Skeleton images already exist. Skipping generation.")

    X_train, X_val, X_test, y_train, y_val, y_test, class_names = prepare_splits()

    # Convert string labels to integers for tf.data
    class_to_idx = {cls: idx for idx, cls in enumerate(class_names)}
    y_train_idx = [class_to_idx[y] for y in y_train]
    y_val_idx   = [class_to_idx[y] for y in y_val]
    y_test_idx  = [class_to_idx[y] for y in y_test]

    train_ds = create_dataset(X_train, y_train_idx, is_training=True)
    val_ds   = create_dataset(X_val, y_val_idx, is_training=False)
    test_ds  = create_dataset(X_test, y_test_idx, is_training=False)

    num_classes = len(class_names)
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6),
    ]

    # Train MobileNetV2
    print("\n--- Training Fast MobileNetV2 ---")
    model_mnv2 = create_fast_model(MobileNetV2, num_classes, model_type='mobilenetv2')
    hist_mnv2 = model_mnv2.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, callbacks=callbacks)
    acc_mnv2 = model_mnv2.evaluate(test_ds)[1]
    print(f"MobileNetV2 Test Accuracy: {acc_mnv2:.4f}")
    model_mnv2.save('mobilenetv2_skeleton.h5')

    # Train EfficientNetB0
    print("\n--- Training Fast EfficientNetB0 ---")
    model_efnb0 = create_fast_model(EfficientNetB0, num_classes, model_type='efficientnet')
    hist_efnb0 = model_efnb0.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, callbacks=callbacks)
    acc_efnb0 = model_efnb0.evaluate(test_ds)[1]
    print(f"EfficientNetB0 Test Accuracy: {acc_efnb0:.4f}")
    model_efnb0.save('efficientnetb0_skeleton.h5')

    # Compare and save best model
    models_list = ['MobileNetV2', 'EfficientNetB0']
    accuracies = [acc_mnv2, acc_efnb0]
    best_idx = int(np.argmax(accuracies))
    
    print(f"\nBest model: {models_list[best_idx]} with accuracy {accuracies[best_idx]:.4f}")
    best_model = model_mnv2 if best_idx == 0 else model_efnb0
    best_model.save('best_skeleton_model.h5')

    # Save class names
    with open('skeleton_class_names.txt', 'w') as f:
        for name in class_names:
            f.write(f"{name}\n")

    # Plot comparison
    plt.figure(figsize=(8, 5))
    bars = plt.bar(models_list, accuracies, color=['blue', 'green'])
    plt.ylabel('Test Accuracy')
    plt.title('Model Comparison on Skeleton Images')
    for bar, acc in zip(bars, accuracies):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01, f'{acc:.3f}', ha='center')
    plt.ylim(0, 1.1)
    plt.tight_layout()
    plt.savefig('skeleton_model_comparison.png')
    plt.show()

    print("All done. Best model saved as 'best_skeleton_model.h5'.")