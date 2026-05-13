import pickle
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2, EfficientNetB0
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

IMG_SIZE = 224
BATCH_SIZE = 32

with open('splits.pkl', 'rb') as f:
    X_train_paths, X_val_paths, X_test_paths, y_train, y_val, y_test = pickle.load(f)

train_df = pd.DataFrame({'filename': X_train_paths, 'class': y_train})
val_df = pd.DataFrame({'filename': X_val_paths, 'class': y_val})
test_df = pd.DataFrame({'filename': X_test_paths, 'class': y_test})

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

# MobileNetV2
model_mnv2 = create_model(MobileNetV2, num_classes)
history_mnv2 = model_mnv2.fit(train_generator, validation_data=val_generator, epochs=20)
test_loss_mnv2, test_acc_mnv2 = model_mnv2.evaluate(test_generator)
print(f"MobileNetV2 test accuracy: {test_acc_mnv2:.4f}")
model_mnv2.save('mobilenetv2_raw.h5')

# EfficientNetB0
model_efnb0 = create_model(EfficientNetB0, num_classes)
history_efnb0 = model_efnb0.fit(train_generator, validation_data=val_generator, epochs=20)
test_loss_efnb0, test_acc_efnb0 = model_efnb0.evaluate(test_generator)
print(f"EfficientNetB0 test accuracy: {test_acc_efnb0:.4f}")
model_efnb0.save('efficientnetb0_raw.h5')