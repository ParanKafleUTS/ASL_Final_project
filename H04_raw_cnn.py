import pickle
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2, EfficientNetB0
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

IMG_SIZE = 224
BATCH_SIZE = 32

# Load train/val/test splits produced by 02_prepare_splits.py
with open('splits.pkl', 'rb') as f:
    X_train_paths, X_val_paths, X_test_paths, y_train, y_val, y_test = pickle.load(f)

train_df = pd.DataFrame({'filename': X_train_paths, 'class': y_train})
val_df   = pd.DataFrame({'filename': X_val_paths,   'class': y_val})
test_df  = pd.DataFrame({'filename': X_test_paths,  'class': y_test})

# ---------------------------------------------------------------------------
# Data generators
# Training: augmented to handle camera noise, lighting shifts, and hand poses.
# Validation / Test: rescale only (no augmentation).
# ---------------------------------------------------------------------------
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    brightness_range=[0.8, 1.2],
    zoom_range=0.1,
    fill_mode='nearest',
)
val_datagen  = ImageDataGenerator(rescale=1.0 / 255)
test_datagen = ImageDataGenerator(rescale=1.0 / 255)

train_generator = train_datagen.flow_from_dataframe(
    train_df, x_col='filename', y_col='class',
    target_size=(IMG_SIZE, IMG_SIZE), batch_size=BATCH_SIZE, class_mode='categorical',
)
val_generator = val_datagen.flow_from_dataframe(
    val_df, x_col='filename', y_col='class',
    target_size=(IMG_SIZE, IMG_SIZE), batch_size=BATCH_SIZE, class_mode='categorical',
)
test_generator = test_datagen.flow_from_dataframe(
    test_df, x_col='filename', y_col='class',
    target_size=(IMG_SIZE, IMG_SIZE), batch_size=BATCH_SIZE, class_mode='categorical',
    shuffle=False,
)

num_classes = len(train_generator.class_indices)
print(f"Number of classes: {num_classes}")

# ---------------------------------------------------------------------------
# Model factory
# Base model weights are frozen for fast initial training via transfer learning.
# ---------------------------------------------------------------------------
def create_model(base_model_cls, num_classes):
    base = base_model_cls(
        weights='imagenet',
        include_top=False,
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
    )
    base.trainable = False  # Freeze pre-trained layers for fast training
    x = base.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.3)(x)
    out = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=base.input, outputs=out)
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy'],
    )
    return model

# ---------------------------------------------------------------------------
# Callbacks: EarlyStopping + ReduceLROnPlateau
# ---------------------------------------------------------------------------
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
callbacks_list = [early_stopping, reduce_lr]

# ---------------------------------------------------------------------------
# MobileNetV2
# ---------------------------------------------------------------------------
print("\n--- Training MobileNetV2 ---")
model_mnv2 = create_model(MobileNetV2, num_classes)
history_mnv2 = model_mnv2.fit(
    train_generator,
    validation_data=val_generator,
    epochs=30,
    callbacks=callbacks_list,
)
test_loss_mnv2, test_acc_mnv2 = model_mnv2.evaluate(test_generator)
print(f"MobileNetV2 test accuracy: {test_acc_mnv2:.4f}")
model_mnv2.save('mobilenetv2_raw.h5')
print("Model saved to mobilenetv2_raw.h5")

# ---------------------------------------------------------------------------
# EfficientNetB0
# ---------------------------------------------------------------------------
print("\n--- Training EfficientNetB0 ---")
model_efnb0 = create_model(EfficientNetB0, num_classes)
history_efnb0 = model_efnb0.fit(
    train_generator,
    validation_data=val_generator,
    epochs=30,
    callbacks=callbacks_list,
)
test_loss_efnb0, test_acc_efnb0 = model_efnb0.evaluate(test_generator)
print(f"EfficientNetB0 test accuracy: {test_acc_efnb0:.4f}")
model_efnb0.save('efficientnetb0_raw.h5')
print("Model saved to efficientnetb0_raw.h5")