import pickle
import pandas as pd
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.layers import RandomRotation, RandomTranslation, RandomZoom, RandomBrightness
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

IMG_SIZE = 224
BATCH_SIZE = 64  # Increased for faster processing

# 1. Load Data
with open('splits.pkl', 'rb') as f:
    X_train_paths, X_val_paths, X_test_paths, y_train, y_val, y_test = pickle.load(f)

# Create label mapping
unique_classes = sorted(list(set(y_train)))
class_to_idx = {cls: idx for idx, cls in enumerate(unique_classes)}
num_classes = len(unique_classes)
print(f"Number of classes: {num_classes}")

# Convert string labels to integers
y_train_idx = [class_to_idx[y] for y in y_train]
y_val_idx = [class_to_idx[y] for y in y_val]
y_test_idx = [class_to_idx[y] for y in y_test]

# 2. Build High-Performance tf.data Pipeline
def process_path(file_path, label):
    # Read and decode image highly efficiently
    img = tf.io.read_file(file_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
    # Note: We do NOT scale to 0-1 here because EfficientNet needs 0-255
    return img, label

AUTOTUNE = tf.data.AUTOTUNE

def create_dataset(paths, labels, is_training=False):
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if is_training:
        ds = ds.shuffle(buffer_size=10000)
    
    # Map the image loading across multiple CPU cores
    ds = ds.map(process_path, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(BATCH_SIZE)
    ds = ds.prefetch(buffer_size=AUTOTUNE) # Loads next batch while current is training
    return ds

train_ds = create_dataset(X_train_paths, y_train_idx, is_training=True)
val_ds   = create_dataset(X_val_paths, y_val_idx, is_training=False)
test_ds  = create_dataset(X_test_paths, y_test_idx, is_training=False)

# 3. Hardware-Accelerated Augmentation inside the Model
data_augmentation = Sequential([
    RandomRotation(0.04),          # roughly 15 degrees
    RandomTranslation(0.1, 0.1),
    RandomZoom(0.1),
    RandomBrightness(0.2)
], name="data_augmentation")

# 4. Model Architecture
def create_fast_model(num_classes):
    inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    
    # Apply augmentations (only runs during training automatically)
    x = data_augmentation(inputs)
    
    # Load EfficientNetB0
    base_model = EfficientNetB0(
        weights='imagenet',
        include_top=False,
        input_tensor=x
    )
    base_model.trainable = False  # Freeze for fast feature extraction
    
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.3)(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs, outputs)
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy', # 'sparse' because labels are integers, not one-hot
        metrics=['accuracy'],
    )
    return model

# 5. Train
callbacks_list = [
    EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6)
]

print("\n--- Training Fast EfficientNetB0 ---")
model_efnb0 = create_fast_model(num_classes)
history_efnb0 = model_efnb0.fit(
    train_ds,
    validation_data=val_ds,
    epochs=30,
    callbacks=callbacks_list,
)

test_loss, test_acc = model_efnb0.evaluate(test_ds)
print(f"EfficientNetB0 test accuracy: {test_acc:.4f}")
model_efnb0.save('efficientnetb0_fast.h5')