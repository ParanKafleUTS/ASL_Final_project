import pickle
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2, EfficientNetB0
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

IMG_SIZE = 224
BATCH_SIZE = 32

with open('crop_splits.pkl', 'rb') as f:
    X_crop_train, X_crop_val, X_crop_test, y_crop_train, y_crop_val, y_crop_test = pickle.load(f)

train_df = pd.DataFrame({'filename': X_crop_train, 'class': y_crop_train})
val_df = pd.DataFrame({'filename': X_crop_val, 'class': y_crop_val})
test_df = pd.DataFrame({'filename': X_crop_test, 'class': y_crop_test})

train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_dataframe(
    train_df, x_col='filename', y_col='class',
    target_size=(IMG_SIZE, IMG_SIZE), batch_size=BATCH_SIZE, class_mode='categorical'
)
val_generator = val_datagen.flow_from_dataframe(
    val_df, x_col='filename', y_col='class',
    target_size=(IMG_SIZE, IMG_SIZE), batch_size=BATCH_SIZE, class_mode='categorical'
)
test_generator = test_datagen.flow_from_dataframe(
    test_df, x_col='filename', y_col='class',
    target_size=(IMG_SIZE, IMG_SIZE), batch_size=BATCH_SIZE, class_mode='categorical', shuffle=False
)

num_classes = len(train_generator.class_indices)

def create_model(base_model, num_classes):
    base = base_model(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
    base.trainable = False
    x = base.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.3)(x)
    out = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=base.input, outputs=out)
    model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# MobileNetV2 on cropped
model_mnv2_crop = create_model(MobileNetV2, num_classes)
history_mnv2_crop = model_mnv2_crop.fit(train_generator, validation_data=val_generator, epochs=20)
test_acc_mnv2_crop = model_mnv2_crop.evaluate(test_generator)[1]
print(f"MobileNetV2 on cropped test accuracy: {test_acc_mnv2_crop:.4f}")
model_mnv2_crop.save('mobilenetv2_crop.h5')

# EfficientNetB0 on cropped
model_efnb0_crop = create_model(EfficientNetB0, num_classes)
history_efnb0_crop = model_efnb0_crop.fit(train_generator, validation_data=val_generator, epochs=20)
test_acc_efnb0_crop = model_efnb0_crop.evaluate(test_generator)[1]
print(f"EfficientNetB0 on cropped test accuracy: {test_acc_efnb0_crop:.4f}")
model_efnb0_crop.save('efficientnetb0_crop.h5')