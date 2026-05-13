import os
import cv2
import numpy as np
import mediapipe as mp
from tqdm import tqdm
import pickle
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2, EfficientNetB0
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import pandas as pd

# ------------------------------
# Configuration
# ------------------------------
SOURCE_DIR = 'subset_5000'                 # original dataset
SKELETON_DIR = 'skeleton_5000'              # output for skeleton images
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 20
TEST_SIZE = 0.15
VAL_SIZE = 0.15
RANDOM_STATE = 42

# Initialize MediaPipe
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)

# ------------------------------
# Step 1: Generate skeleton images
# ------------------------------
def generate_skeleton_images():
    """Create skeleton images from original dataset and save to SKELETON_DIR."""
    os.makedirs(SKELETON_DIR, exist_ok=True)

    # Get all class folders
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

            # Skip if already exists
            if os.path.exists(dst_path):
                total += 1
                continue

            # Read image
            image = cv2.imread(src_path)
            if image is None:
                print(f"Warning: cannot read {src_path}")
                skipped += 1
                continue

            # Detect hand
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = hands.process(image_rgb)

            if results.multi_hand_landmarks:
                # Create white background
                skeleton = np.ones((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8) * 255

                # Draw landmarks and connections
                for hand_landmarks in results.multi_hand_landmarks:
                    # We need to scale landmarks to image size (0..IMG_SIZE)
                    # MediaPipe landmarks are normalized [0,1], so multiply by IMG_SIZE
                    h, w, _ = skeleton.shape
                    # mp_draw drawing utilities require landmarks in the same format,
                    # but they draw on the original image coordinates. To draw on our blank image,
                    # we need to scale and draw manually or create a temporary copy.
                    # Simpler: draw on a blank image using scaled coordinates.
                    # We'll manually draw circles and lines.

                    # Extract scaled coordinates
                    points = []
                    for lm in hand_landmarks.landmark:
                        x = int(lm.x * IMG_SIZE)
                        y = int(lm.y * IMG_SIZE)
                        points.append((x, y))

                    # Draw connections (using MediaPipe's HAND_CONNECTIONS)
                    for connection in mp_hands.HAND_CONNECTIONS:
                        start_idx, end_idx = connection
                        cv2.line(skeleton, points[start_idx], points[end_idx], (0, 0, 0), 2)

                    # Draw landmark circles
                    for (x, y) in points:
                        cv2.circle(skeleton, (x, y), 3, (0, 0, 255), -1)  # red dots

                # Save skeleton image
                cv2.imwrite(dst_path, skeleton)
                total += 1
            else:
                skipped += 1

    print(f"Skeleton generation complete. Total processed: {total}, Skipped (no hand): {skipped}")

# ------------------------------
# Step 2: Prepare data splits
# ------------------------------
def prepare_splits():
    """Gather all skeleton image paths and labels, split into train/val/test."""
    image_paths = []
    labels = []
    classes = [d for d in os.listdir(SKELETON_DIR) if os.path.isdir(os.path.join(SKELETON_DIR, d))]
    for cls in classes:
        class_path = os.path.join(SKELETON_DIR, cls)
        images = [os.path.join(class_path, f) for f in os.listdir(class_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        image_paths.extend(images)
        labels.extend([cls] * len(images))

    # First split: training + temp (validation + test)
    X_train, X_temp, y_train, y_temp = train_test_split(
        image_paths, labels, test_size=(VAL_SIZE + TEST_SIZE), stratify=labels, random_state=RANDOM_STATE
    )
    # Split temp into validation and test
    val_ratio = VAL_SIZE / (VAL_SIZE + TEST_SIZE)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=(1 - val_ratio), stratify=y_temp, random_state=RANDOM_STATE
    )

    print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    return X_train, X_val, X_test, y_train, y_val, y_test

# ------------------------------
# Step 3: Create model
# ------------------------------
def create_model(base_model_class, num_classes):
    base = base_model_class(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
    base.trainable = False
    x = base.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.3)(x)
    out = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=base.input, outputs=out)
    model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# ------------------------------
# Step 4: Train and evaluate
# ------------------------------
def train_and_evaluate(model, train_gen, val_gen, test_gen, model_name):
    print(f"\n--- Training {model_name} ---")
    history = model.fit(train_gen, validation_data=val_gen, epochs=EPOCHS)
    test_loss, test_acc = model.evaluate(test_gen)
    print(f"{model_name} Test Accuracy: {test_acc:.4f}")
    return model, test_acc, history

# ------------------------------
# Main execution
# ------------------------------
if __name__ == "__main__":
    # Step 1: generate skeletons (if not already done)
    if not os.path.exists(SKELETON_DIR) or len(os.listdir(SKELETON_DIR)) == 0:
        print("Generating skeleton images...")
        generate_skeleton_images()
    else:
        print("Skeleton images already exist. Skipping generation.")

    # Step 2: prepare splits
    X_train, X_val, X_test, y_train, y_val, y_test = prepare_splits()

    # Save splits for later use
    with open('skeleton_splits.pkl', 'wb') as f:
        pickle.dump((X_train, X_val, X_test, y_train, y_val, y_test), f)

    # Create dataframes for generators
    train_df = pd.DataFrame({'filename': X_train, 'class': y_train})
    val_df = pd.DataFrame({'filename': X_val, 'class': y_val})
    test_df = pd.DataFrame({'filename': X_test, 'class': y_test})

    # Data generators (no augmentation, just rescale)
    datagen = ImageDataGenerator(rescale=1./255)
    train_gen = datagen.flow_from_dataframe(train_df, x_col='filename', y_col='class',
                                            target_size=(IMG_SIZE, IMG_SIZE), batch_size=BATCH_SIZE,
                                            class_mode='categorical')
    val_gen = datagen.flow_from_dataframe(val_df, x_col='filename', y_col='class',
                                          target_size=(IMG_SIZE, IMG_SIZE), batch_size=BATCH_SIZE,
                                          class_mode='categorical')
    test_gen = datagen.flow_from_dataframe(test_df, x_col='filename', y_col='class',
                                           target_size=(IMG_SIZE, IMG_SIZE), batch_size=BATCH_SIZE,
                                           class_mode='categorical', shuffle=False)

    num_classes = len(train_gen.class_indices)

    # Step 4: Train MobileNetV2
    model_mnv2 = create_model(MobileNetV2, num_classes)
    model_mnv2, acc_mnv2, hist_mnv2 = train_and_evaluate(model_mnv2, train_gen, val_gen, test_gen, "MobileNetV2")
    model_mnv2.save('mobilenetv2_skeleton.h5')

    # Train EfficientNetB0
    model_efnb0 = create_model(EfficientNetB0, num_classes)
    model_efnb0, acc_efnb0, hist_efnb0 = train_and_evaluate(model_efnb0, train_gen, val_gen, test_gen, "EfficientNetB0")
    model_efnb0.save('efficientnetb0_skeleton.h5')

    # Step 5: Compare and save best
    models = ['MobileNetV2', 'EfficientNetB0']
    accuracies = [acc_mnv2, acc_efnb0]
    best_idx = np.argmax(accuracies)
    best_model_name = models[best_idx]
    best_accuracy = accuracies[best_idx]
    best_model = model_mnv2 if best_idx == 0 else model_efnb0

    print(f"\nBest model: {best_model_name} with accuracy {best_accuracy:.4f}")
    best_model.save('best_skeleton_model.h5')

    # Save class names
    class_names = [name for name, idx in sorted(train_gen.class_indices.items(), key=lambda item: item[1])]
    with open('skeleton_class_names.txt', 'w') as f:
        for name in class_names:
            f.write(f"{name}\n")

    # Plot comparison
    plt.figure(figsize=(8,5))
    bars = plt.bar(models, accuracies, color=['blue', 'green'])
    plt.ylabel('Test Accuracy')
    plt.title('Model Comparison on Skeleton Images')
    for bar, acc in zip(bars, accuracies):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f'{acc:.3f}', ha='center')
    plt.ylim(0, 1.1)
    plt.savefig('skeleton_model_comparison.png')
    plt.show()

    print("All done. Best model saved as 'best_skeleton_model.h5'.")