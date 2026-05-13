import pickle
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pandas as pd

# Load one of the generators to get class indices
with open('crop_splits.pkl', 'rb') as f:
    X_crop_train, _, _, y_crop_train, _, _ = pickle.load(f)

train_df = pd.DataFrame({'filename': X_crop_train, 'class': y_crop_train})
train_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_dataframe(
    train_df, x_col='filename', y_col='class',
    target_size=(224,224), batch_size=32, class_mode='categorical'
)

class_indices = train_generator.class_indices
# Invert to get list of class names in order of index
class_names = [None] * len(class_indices)
for name, idx in class_indices.items():
    class_names[idx] = name

with open('class_names.txt', 'w') as f:
    for name in class_names:
        f.write(f"{name}\n")

print("Class names saved.")