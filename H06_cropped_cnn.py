import pickle
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2, EfficientNetB0
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, Rescaling
from tensorflow.keras.layers import RandomRotation, RandomTranslation, RandomZoom, RandomBrightness
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

IMG_SIZE = 224
BATCH_SIZE = 64  # Increased batch size because tf.data is much more efficient

# ---------------------------------------------------------------------------
# 1. Load splits and encode labels as integers (required for fast tf.data)
# ---------------------------------------------------------------------------
with open('processed_splits.pkl', 'rb') as f:
    X_crop_train, X_crop_val, X_crop_test, y_crop_train, y_crop_val, y_crop_test = pickle.load(f)

# Get unique classes and create a mapping to integers
unique_classes = sorted(list(set(y_crop_train)))
class_to_idx = {cls: idx for idx, cls in enumerate(unique_classes)}
num_classes = len(unique_classes)
print(f"Number of classes: {num_classes}")

# Convert string labels to integers
y_train_idx = [class_to_idx[y] for y in y_crop_train]
y_val_idx = [class_to_idx[y] for y in y_crop_train] # Wait, typo fix below
y_val_idx   = [class_to_idx[y] for y in y_crop_val]
y_test_idx  = [class_to_idx[y] for y in y_crop_test]

# ---------------------------------------------------------------------------
# 2. Build High-Performance tf.data Pipeline
# ---------------------------------------------------------------------------
AUTOTUNE = tf.data.AUTOTUNE

def process_path(file_path, label):
    # Read and decode image using optimized C++ operations
    img = tf.io.read_file(file_path)
    img = tf.image.decode_jpeg(img, channels=3) # Use decode_png if your images are PNGs
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
    return img, label

def create_dataset(paths, labels, is_training=False):
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if is_training:
        ds = ds.shuffle(buffer_size=10000)
    
    # Map the image loading across multiple CPU cores
    ds = ds.map(process_path, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(BATCH_SIZE)
    ds = ds.prefetch(buffer_size=AUTOTUNE) # Loads next batch while current is training on GPU
    return ds

train_ds = create_dataset(X_crop_train, y_train_idx, is_training=True)
val_ds   = create_dataset(X_crop_val, y_val_idx, is_training=False)
test_ds  = create_dataset(X_crop_test, y_test_idx, is_training=False)

# ---------------------------------------------------------------------------
# 3. Hardware-Accelerated Augmentation (Runs on GPU)
# ---------------------------------------------------------------------------
data_augmentation = Sequential([
    RandomRotation(0.04),          # roughly 15 degrees
    RandomTranslation(0.1, 0.1),
    RandomZoom(0.1),
    RandomBrightness(0.2)
], name="data_augmentation")

# ---------------------------------------------------------------------------
# 4. Model Factory (Handles Scaling internally per-model)
# ---------------------------------------------------------------------------
def create_fast_model(base_model_cls, num_classes, model_type):
    inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    
    # Apply augmentations (only runs during training, automatically disabled in val/test)
    x = data_augmentation(inputs)
    
    # MobileNetV2 expects [-1, 1], so we add a scaler layer
    if model_type == 'mobilenetv2':
        x = Rescaling(scale=1./127.5, offset=-1)(x)
    # EfficientNetB0 expects [0, 255] and handles it internally, so we don't scale!
    
    base_model = base_model_cls(
        weights='imagenet',
        include_top=False,
        input_tensor=x
    )
    base_model.trainable = False  # Freeze pre-trained layers
    
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.3)(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs, outputs)
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy', # 'sparse' because labels are ints, not one-hot
        metrics=['accuracy'],
    )
    return model

# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------
callbacks_list = [
    EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6)
]

# ===========================================================================
# Train MobileNetV2
# ===========================================================================
print("\n--- Training Fast MobileNetV2 on cropped images ---")
model_mnv2_crop = create_fast_model(MobileNetV2, num_classes, model_type='mobilenetv2')
history_mnv2_crop = model_mnv2_crop.fit(
    train_ds,
    validation_data=val_ds,
    epochs=30,
    callbacks=callbacks_list,
)
test_acc_mnv2 = model_mnv2_crop.evaluate(test_ds)[1]
print(f"MobileNetV2 (cropped) test accuracy: {test_acc_mnv2:.4f}")
model_mnv2_crop.save('mobilenetv2_crop.h5')


# ===========================================================================
# Train EfficientNetB0
# ===========================================================================
print("\n--- Training Fast EfficientNetB0 on cropped images ---")
model_efnb0_crop = create_fast_model(EfficientNetB0, num_classes, model_type='efficientnet')
history_efnb0_crop = model_efnb0_crop.fit(
    train_ds,
    validation_data=val_ds,
    epochs=30,
    callbacks=callbacks_list,
)
test_acc_efnb0 = model_efnb0_crop.evaluate(test_ds)[1]
print(f"EfficientNetB0 (cropped) test accuracy: {test_acc_efnb0:.4f}")
model_efnb0_crop.save('efficientnetb0_crop.h5')