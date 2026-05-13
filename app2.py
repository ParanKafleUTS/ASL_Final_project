"""
Transfer Learning for Camera-Specific Hand Gesture Recognition
Fine-tunes the MobileNetV2 model on camera-specific data collected live
"""

import os
import cv2
import numpy as np
import logging
from pathlib import Path
from datetime import datetime
import pickle

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split

# ===============================
# LOGGING SETUP
# ===============================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ===============================
# CONFIGURATION
# ===============================
class Config:
    """Configuration for transfer learning."""
    # Paths
    ORIGINAL_MODEL = "mobilenetv2_crop.h5"
    CLASS_NAMES_PATH = "class_names.txt"
    CAMERA_DATA_DIR = "camera_training_data"  # Directory to collect camera-specific data
    RETRAINED_MODEL = "mobilenetv2_crop_finetuned.h5"
    
    # Image settings
    IMG_SIZE = 224
    
    # Training settings
    EPOCHS = 5
    BATCH_SIZE = 16
    LEARNING_RATE = 0.0001  # Low learning rate for fine-tuning
    TRAIN_TEST_SPLIT = 0.8
    VAL_SPLIT = 0.2
    
    # Layer freezing
    FREEZE_UNTIL_LAYER = "block_13_project"  # Freeze up to this layer
    
    # Data augmentation
    AUGMENTATION = True


# ===============================
# DATA COLLECTION
# ===============================

def create_camera_data_structure(base_dir: str, class_names: list):
    """Create directory structure for collecting camera-specific data."""
    logger.info(f"Creating directory structure in {base_dir}")
    
    os.makedirs(base_dir, exist_ok=True)
    
    for class_name in class_names:
        class_dir = os.path.join(base_dir, class_name)
        os.makedirs(class_dir, exist_ok=True)
    
    logger.info(f"✓ Created directories for {len(class_names)} classes")
    return base_dir


def collect_camera_data_interactive(
    camera_data_dir: str,
    class_names: list,
    samples_per_class: int = 30,
    img_size: int = 224
):
    """
    Interactively collect hand gesture images from camera for transfer learning.
    
    Press:
    - SPACE: Capture image for current class
    - N: Move to next class
    - Q: Quit
    """
    import mediapipe as mp
    
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.5
    )
    mp_draw = mp.solutions.drawing_utils
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logger.error("Cannot open camera")
        return False
    
    logger.info("Starting interactive data collection...")
    logger.info(f"Collect {samples_per_class} images per class")
    logger.info("Controls: SPACE=capture, N=next class, Q=quit")
    
    for class_idx, class_name in enumerate(class_names):
        class_dir = os.path.join(camera_data_dir, class_name)
        captured_count = 0
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Class {class_idx + 1}/{len(class_names)}: {class_name}")
        logger.info(f"Capture {samples_per_class} images - Show gesture to camera")
        logger.info(f"{'='*60}")
        
        while captured_count < samples_per_class:
            success, frame = cap.read()
            if not success:
                logger.error("Failed to read frame")
                break
            
            frame = cv2.flip(frame, 1)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Detect hands
            results = hands.process(frame_rgb)
            
            # Draw on frame
            h, w, c = frame.shape
            status_text = f"{class_name} - Captured: {captured_count}/{samples_per_class}"
            cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                       1, (0, 255, 0), 2)
            cv2.putText(frame, "SPACE=Capture | N=Next | Q=Quit", (10, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                # Draw bounding box
                xs = [lm.x * w for lm in hand_landmarks.landmark]
                ys = [lm.y * h for lm in hand_landmarks.landmark]
                x_min, x_max = int(min(xs)), int(max(xs))
                y_min, y_max = int(min(ys)), int(max(ys))
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            
            cv2.imshow(f"Collecting: {class_name}", frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord(" "):  # SPACE to capture
                if results.multi_hand_landmarks:
                    hand_landmarks = results.multi_hand_landmarks[0]
                    
                    # Crop hand region
                    xs = [lm.x * w for lm in hand_landmarks.landmark]
                    ys = [lm.y * h for lm in hand_landmarks.landmark]
                    x_min, x_max = int(min(xs)), int(max(xs))
                    y_min, y_max = int(min(ys)), int(max(ys))
                    
                    margin = int((x_max - x_min) * 0.2)
                    x_min = max(0, x_min - margin)
                    y_min = max(0, y_min - margin)
                    x_max = min(w, x_max + margin)
                    y_max = min(h, y_max + margin)
                    
                    crop = frame[y_min:y_max, x_min:x_max]
                    
                    if crop.size > 0:
                        # Resize
                        crop_resized = cv2.resize(crop, (img_size, img_size))
                        
                        # Save
                        timestamp = int(datetime.now().timestamp() * 1000)
                        filename = f"{class_name}_{captured_count}_{timestamp}.jpg"
                        filepath = os.path.join(class_dir, filename)
                        
                        cv2.imwrite(filepath, crop_resized)
                        captured_count += 1
                        logger.info(f"✓ Captured {captured_count}/{samples_per_class}: {filename}")
                else:
                    logger.warning("No hand detected in frame")
            
            elif key == ord("n"):  # N to next class
                break
            
            elif key == ord("q"):  # Q to quit
                cap.release()
                cv2.destroyAllWindows()
                logger.info("Data collection cancelled")
                return False
        
        if captured_count < samples_per_class:
            logger.warning(f"Only captured {captured_count}/{samples_per_class} for {class_name}")
    
    cap.release()
    cv2.destroyAllWindows()
    logger.info("✓ Data collection complete!")
    return True


# ===============================
# TRANSFER LEARNING
# ===============================

def load_class_names(filepath: str) -> list:
    """Load class names from file."""
    with open(filepath, 'r') as f:
        return [line.strip() for line in f.readlines() if line.strip()]


def prepare_data_generators(data_dir: str, img_size: int, batch_size: int, 
                           augmentation: bool = True):
    """
    Prepare training and validation data generators.
    """
    logger.info(f"Preparing data from {data_dir}")
    
    if augmentation:
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=15,
            width_shift_range=0.1,
            height_shift_range=0.1,
            zoom_range=0.2,
            horizontal_flip=True,
            brightness_range=[0.8, 1.2],
            shear_range=0.15
        )
    else:
        train_datagen = ImageDataGenerator(rescale=1./255)
    
    val_datagen = ImageDataGenerator(rescale=1./255)
    
    train_generator = train_datagen.flow_from_directory(
        data_dir,
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode='categorical',
        subset='training' if augmentation else None,
        shuffle=True
    )
    
    val_generator = val_datagen.flow_from_directory(
        data_dir,
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation' if augmentation else None,
        shuffle=False
    )
    
    logger.info(f"✓ Loaded training samples: {train_generator.samples}")
    logger.info(f"✓ Loaded validation samples: {val_generator.samples}")
    
    return train_generator, val_generator


def freeze_base_model(model, freeze_until_layer: str = None):
    """
    Freeze early layers to preserve learned features from original training.
    Only train the last few layers on camera-specific data.
    """
    if freeze_until_layer is None:
        # Freeze all but last 20 layers
        for layer in model.layers[:-20]:
            layer.trainable = False
        logger.info(f"✓ Froze {len(model.layers) - 20} layers, training last 20")
    else:
        # Freeze until specified layer
        freeze_flag = True
        for layer in model.layers:
            if freeze_until_layer in layer.name:
                freeze_flag = False
            layer.trainable = not freeze_flag
        
        trainable_count = sum(1 for layer in model.layers if layer.trainable)
        logger.info(f"✓ Froze layers up to '{freeze_until_layer}', training {trainable_count} layers")


def fine_tune_model(
    model_path: str,
    train_generator,
    val_generator,
    epochs: int = 5,
    learning_rate: float = 0.0001,
    freeze_until_layer: str = None
):
    """
    Fine-tune the model on camera-specific data.
    """
    logger.info("Loading original model...")
    model = load_model(model_path)
    
    # Freeze base layers
    freeze_base_model(model, freeze_until_layer)
    
    # Compile with low learning rate
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    logger.info(f"Starting fine-tuning for {epochs} epochs...")
    
    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=epochs,
        verbose=1,
        steps_per_epoch=len(train_generator),
        validation_steps=len(val_generator)
    )
    
    return model, history


def evaluate_model(model, test_generator):
    """Evaluate model on test data."""
    logger.info("Evaluating model on test data...")
    
    loss, accuracy = model.evaluate(test_generator, verbose=1)
    
    logger.info(f"Test Loss: {loss:.4f}")
    logger.info(f"Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    return loss, accuracy


# ===============================
# MAIN PIPELINE
# ===============================

def main():
    """Main transfer learning pipeline."""
    config = Config()
    
    logger.info("="*70)
    logger.info("Hand Gesture Recognition - Transfer Learning Pipeline")
    logger.info("="*70)
    
    # Step 1: Load class names
    logger.info("\n[1/4] Loading class names...")
    class_names = load_class_names(config.CLASS_NAMES_PATH)
    logger.info(f"✓ Loaded {len(class_names)} classes: {class_names[:5]}...")
    
    # Step 2: Create directory structure and collect data
    logger.info("\n[2/4] Setting up data collection...")
    create_camera_data_structure(config.CAMERA_DATA_DIR, class_names)
    
    logger.info("\nCollecting camera-specific training data...")
    logger.info("Make sure you have good lighting and clear hand gestures")
    
    success = collect_camera_data_interactive(
        config.CAMERA_DATA_DIR,
        class_names,
        samples_per_class=30,
        img_size=config.IMG_SIZE
    )
    
    if not success:
        logger.error("Data collection failed or cancelled")
        return
    
    # Step 3: Prepare data and fine-tune
    logger.info("\n[3/4] Preparing data generators...")
    train_generator, val_generator = prepare_data_generators(
        config.CAMERA_DATA_DIR,
        config.IMG_SIZE,
        config.BATCH_SIZE,
        config.AUGMENTATION
    )
    
    logger.info("\n[4/4] Fine-tuning model on camera-specific data...")
    model, history = fine_tune_model(
        config.ORIGINAL_MODEL,
        train_generator,
        val_generator,
        epochs=config.EPOCHS,
        learning_rate=config.LEARNING_RATE,
        freeze_until_layer=config.FREEZE_UNTIL_LAYER
    )
    
    # Step 5: Save the fine-tuned model
    logger.info(f"\nSaving fine-tuned model to {config.RETRAINED_MODEL}...")
    model.save(config.RETRAINED_MODEL)
    logger.info(f"✓ Model saved!")
    
    # Final evaluation
    logger.info("\n" + "="*70)
    logger.info("Fine-tuning complete!")
    logger.info(f"Original model: {config.ORIGINAL_MODEL}")
    logger.info(f"Fine-tuned model: {config.RETRAINED_MODEL}")
    logger.info("="*70)
    
    logger.info("\n📌 Next steps:")
    logger.info(f"1. Update app.py to use: MODEL_PATH = '{config.RETRAINED_MODEL}'")
    logger.info("2. Run: python app.py")


if __name__ == "__main__":
    main()