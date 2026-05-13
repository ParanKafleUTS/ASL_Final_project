"""
Hand Gesture Recognition - Live Video Application (Skeleton Model)
with sentence building, auto‑correct, and thumbs‑up acceptance.
"""

import os
import sys
import logging
import time
from collections import deque
from typing import Optional, Tuple, List

import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
from spellchecker import SpellChecker

# ===============================
# LOGGING CONFIGURATION
# ===============================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('gesture_recognition.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# ===============================
# CONFIGURATION
# ===============================
class Config:
    """Configuration constants for the gesture recognition pipeline."""
    IMG_SIZE = 224
    SAVE_DIR = "live_skeletons"
    MODEL_PATH = "best_skeleton_model.h5"
    CLASS_NAMES_PATH = "skeleton_class_names.txt"

    # Hand detection parameters
    MIN_DETECTION_CONFIDENCE = 0.5
    MAX_HANDS = 1

    # Movement detection parameters (optional, can be disabled)
    MOVEMENT_THRESHOLD = 15
    STABLE_DURATION = 1.0
    PREDICTION_INTERVAL = 0.5
    USE_MOVEMENT_DETECTION = True   # set to False if you want continuous prediction

    # Prediction parameters
    PREDICTION_CONFIDENCE = 0.7
    PREDICTION_BUFFER = 5
    CAMERA_INDEX = 0

    DEBUG_MODE = True
    SAVE_SKELETONS = True


def validate_file_exists(filepath: str, file_type: str = "file") -> bool:
    """Validate that a required file exists."""
    if not os.path.exists(filepath):
        logger.error(f"{file_type} not found: {filepath}")
        return False
    return True


def load_class_names(filepath: str) -> Optional[List[str]]:
    """Load class names from a text file."""
    if not validate_file_exists(filepath, "Class names file"):
        return None

    try:
        with open(filepath, "r") as f:
            class_names = [line.strip() for line in f.readlines() if line.strip()]

        if not class_names:
            logger.error("Class names file is empty")
            return None

        logger.info(f"Loaded {len(class_names)} class names")
        return class_names

    except Exception as e:
        logger.error(f"Failed to load class names: {e}")
        return None


def load_gesture_model(model_path: str):
    """Load the trained gesture recognition model."""
    if not validate_file_exists(model_path, "Model file"):
        return None

    try:
        model = load_model(model_path)
        logger.info(f"Model loaded: input {model.input_shape}, output {model.output_shape}")
        return model
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return None


def initialize_camera(camera_index: int = 0) -> Optional[cv2.VideoCapture]:
    """Initialize and verify camera connection."""
    try:
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            logger.error("Failed to open camera")
            return None

        logger.info(f"Camera initialized")
        return cap
    except Exception as e:
        logger.error(f"Camera initialization error: {e}")
        return None


def initialize_mediapipe_hands(
    static_mode: bool = False,
    max_hands: int = 1,
    min_confidence: float = 0.5
) -> Tuple[object, object, object]:
    """Initialize MediaPipe hand detection pipeline."""
    try:
        mp_hands = mp.solutions.hands
        hands = mp_hands.Hands(
            static_image_mode=static_mode,
            max_num_hands=max_hands,
            min_detection_confidence=min_confidence,
            min_tracking_confidence=0.5
        )
        mp_draw = mp.solutions.drawing_utils
        logger.info("MediaPipe hands initialized")
        return mp_hands, hands, mp_draw
    except Exception as e:
        logger.error(f"MediaPipe initialization error: {e}")
        return None, None, None


def create_output_directory(directory: str) -> bool:
    """Create output directory if it doesn't exist."""
    try:
        os.makedirs(directory, exist_ok=True)
        return True
    except Exception as e:
        logger.error(f"Failed to create output directory: {e}")
        return False


def is_thumbs_up(landmarks) -> bool:
    """
    Heuristic to detect thumbs‑up gesture.
    Thumb tip is above thumb IP, and all other fingers are curled.
    """
    thumb_tip = landmarks.landmark[4]
    thumb_ip = landmarks.landmark[3]
    index_tip = landmarks.landmark[8]
    middle_tip = landmarks.landmark[12]
    ring_tip = landmarks.landmark[16]
    pinky_tip = landmarks.landmark[20]

    thumb_up = thumb_tip.y < thumb_ip.y
    fingers_curled = (index_tip.y > landmarks.landmark[6].y and
                      middle_tip.y > landmarks.landmark[10].y and
                      ring_tip.y > landmarks.landmark[14].y and
                      pinky_tip.y > landmarks.landmark[18].y)
    return thumb_up and fingers_curled


def generate_skeleton_image(landmarks, img_size: int = 224) -> np.ndarray:
    """
    Create a white image with hand skeleton drawn.
    - Landmarks are converted to pixel coordinates.
    - Connections drawn in black.
    - Landmark points drawn as red circles.
    """
    skeleton = np.ones((img_size, img_size, 3), dtype=np.uint8) * 255

    points = []
    for lm in landmarks.landmark:
        x = int(lm.x * img_size)
        y = int(lm.y * img_size)
        points.append((x, y))

    for connection in mp.solutions.hands.HAND_CONNECTIONS:
        start_idx, end_idx = connection
        cv2.line(skeleton, points[start_idx], points[end_idx], (0, 0, 0), 2)

    for (x, y) in points:
        cv2.circle(skeleton, (x, y), 3, (0, 0, 255), -1)

    return skeleton


def detect_movement(
    prev_gray: Optional[np.ndarray],
    curr_gray: np.ndarray
) -> float:
    """Calculate movement magnitude between consecutive frames."""
    try:
        if prev_gray is None or prev_gray.shape != curr_gray.shape:
            return 0.0

        diff = cv2.absdiff(prev_gray, curr_gray)
        diff_sum = np.sum(diff) / (curr_gray.shape[0] * curr_gray.shape[1])
        return diff_sum
    except Exception as e:
        logger.error(f"Error in detect_movement: {e}")
        return 0.0


def predict_gesture(
    model,
    skeleton_img: np.ndarray,
    class_names: List[str],
    confidence_threshold: float = 0.5,
    debug: bool = False
) -> Tuple[str, float, np.ndarray]:
    """Perform gesture prediction on skeleton image."""
    try:
        if model is None or skeleton_img is None:
            return "Unknown", 0.0, np.array([])

        if skeleton_img.shape != (224, 224, 3):
            return "Unknown", 0.0, np.array([])

        img_norm = skeleton_img.astype(np.float32) / 255.0
        input_tensor = np.expand_dims(img_norm, axis=0)

        predictions = model.predict(input_tensor, verbose=0)[0]

        if predictions.shape[0] != len(class_names):
            logger.error(f"Model output mismatch")
            return "Unknown", 0.0, predictions

        class_idx = np.argmax(predictions)
        confidence = float(predictions[class_idx])
        predicted_class = class_names[class_idx]

        if debug:
            top_indices = np.argsort(predictions)[-5:][::-1]
            logger.info("🎯 Top 5 predictions:")
            for idx in top_indices:
                bar_length = int(predictions[idx] * 30)
                bar = "█" * bar_length
                logger.info(f"   {class_names[idx]:8s} {predictions[idx]:.3f} {bar}")

        if confidence < confidence_threshold:
            return "Unknown", confidence, predictions

        return predicted_class, confidence, predictions

    except Exception as e:
        logger.error(f"Error in predict_gesture: {e}")
        return "Unknown", 0.0, np.array([])


def smooth_predictions(
    prediction_queue: deque,
    current_prediction: str
) -> str:
    """Smooth predictions using voting over a buffer."""
    try:
        prediction_queue.append(current_prediction)
        if len(prediction_queue) == 0:
            return "Unknown"

        counts = {}
        for pred in prediction_queue:
            counts[pred] = counts.get(pred, 0) + 1

        most_common = max(counts, key=counts.get)
        return most_common
    except Exception as e:
        logger.error(f"Error smoothing predictions: {e}")
        return "Unknown"


def save_skeleton(skeleton: np.ndarray, label: str, output_dir: str) -> Optional[str]:
    """Save skeleton image to disk with label."""
    try:
        if skeleton is None or skeleton.size == 0:
            return None

        timestamp = int(time.time() * 1000)
        save_path = os.path.join(output_dir, f"{label}_{timestamp}.jpg")

        success = cv2.imwrite(save_path, skeleton)
        return save_path if success else None

    except Exception as e:
        logger.error(f"Error saving skeleton: {e}")
        return None


# ===============================
# MAIN APPLICATION CLASS
# ===============================

class GestureRecognitionApp:
    """Main application for real-time hand gesture recognition with sentence building."""

    def __init__(self, config: Config):
        """Initialize the gesture recognition application."""
        self.config = config
        self.model = None
        self.class_names = None
        self.mp_hands = None
        self.hands = None
        self.mp_draw = None
        self.cap = None
        self.spell = SpellChecker()

        # Movement detection
        self.prev_crop_gray = None
        self.last_move_time = time.time()
        self.last_prediction_time = time.time()
        self.frame_count = 0
        self.prediction_queue = deque(maxlen=config.PREDICTION_BUFFER)

        # Sentence building
        self.sentence = ""
        self.current_word = ""

        logger.info("Initializing Hand Gesture Recognition (Skeleton Model with Sentence Building)")

    def validate_setup(self) -> bool:
        """Validate all required files exist."""
        logger.info("Validating setup...")

        if not validate_file_exists(self.config.MODEL_PATH, "Model"):
            return False
        if not validate_file_exists(self.config.CLASS_NAMES_PATH, "Class names"):
            return False
        if not create_output_directory(self.config.SAVE_DIR):
            return False

        logger.info("✓ Setup validation passed")
        return True

    def initialize(self) -> bool:
        """Initialize all components."""
        logger.info("Initializing components...")

        self.model = load_gesture_model(self.config.MODEL_PATH)
        if self.model is None:
            return False

        self.class_names = load_class_names(self.config.CLASS_NAMES_PATH)
        if self.class_names is None:
            return False

        self.mp_hands, self.hands, self.mp_draw = initialize_mediapipe_hands(
            static_mode=False,
            max_hands=self.config.MAX_HANDS,
            min_confidence=self.config.MIN_DETECTION_CONFIDENCE
        )
        if self.hands is None:
            return False

        self.cap = initialize_camera(self.config.CAMERA_INDEX)
        if self.cap is None:
            return False

        logger.info("✓ All components initialized")
        return True

    def process_frame(self, frame: np.ndarray):
        """Process a single video frame for gesture recognition and update display."""
        self.frame_count += 1

        try:
            frame = cv2.flip(frame, 1)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            results = self.hands.process(frame_rgb)

            if not results.multi_hand_landmarks:
                self.prev_crop_gray = None
                self.last_move_time = time.time()
                self.prediction_queue.clear()
                return

            hand_landmarks = results.multi_hand_landmarks[0]
            self.mp_draw.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

            # Check for thumbs‑up gesture to accept current word
            if is_thumbs_up(hand_landmarks):
                if self.current_word:
                    corrected = self.spell.correction(self.current_word)
                    if corrected:
                        self.sentence += corrected + " "
                    else:
                        self.sentence += self.current_word + " "
                    self.current_word = ""
                    self.prediction_queue.clear()
                # Skip prediction while accepting
                return

            # ------------------ Movement detection (optional) ------------------
            if self.config.USE_MOVEMENT_DETECTION:
                # For movement, we use a crop from the original frame (grayscale)
                h, w, _ = frame.shape
                xs = [lm.x * w for lm in hand_landmarks.landmark]
                ys = [lm.y * h for lm in hand_landmarks.landmark]
                x_min, x_max = int(min(xs)), int(max(xs))
                y_min, y_max = int(min(ys)), int(max(ys))
                padding_x = int((x_max - x_min) * 0.2)
                padding_y = int((y_max - y_min) * 0.2)
                x_min = max(0, x_min - padding_x)
                y_min = max(0, y_min - padding_y)
                x_max = min(w, x_max + padding_x)
                y_max = min(h, y_max + padding_y)
                crop = frame[y_min:y_max, x_min:x_max]
                if crop.size == 0:
                    self.prev_crop_gray = None
                    self.last_move_time = time.time()
                    return

                crop_gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                movement = detect_movement(self.prev_crop_gray, crop_gray)
                self.prev_crop_gray = crop_gray.copy()

                # If movement above threshold, reset stable timer and prediction queue
                if movement >= self.config.MOVEMENT_THRESHOLD:
                    self.last_move_time = time.time()
                    self.prediction_queue.clear()
                    return

                time_stable = time.time() - self.last_move_time
                if time_stable < self.config.STABLE_DURATION:
                    return

            # ------------------ Generate skeleton and predict ------------------
            skeleton = generate_skeleton_image(hand_landmarks, self.config.IMG_SIZE)

            # Only predict at intervals
            time_since_last_pred = time.time() - self.last_prediction_time
            if time_since_last_pred > self.config.PREDICTION_INTERVAL:
                predicted_class, confidence, _ = predict_gesture(
                    self.model,
                    skeleton,
                    self.class_names,
                    self.config.PREDICTION_CONFIDENCE,
                    debug=self.config.DEBUG_MODE
                )

                if predicted_class != "Unknown":
                    # Smooth prediction over buffer
                    smoothed = smooth_predictions(self.prediction_queue, predicted_class)
                    if smoothed != "Unknown":
                        # Append letter to current word
                        self.current_word += smoothed
                        logger.info(f"✋ Added letter '{smoothed}' -> word: '{self.current_word}'")

                self.last_prediction_time = time.time()

                # Save skeleton if enabled
                if self.config.SAVE_SKELETONS and predicted_class != "Unknown":
                    save_skeleton(skeleton, predicted_class, self.config.SAVE_DIR)

            # Show the skeleton in a separate window
            cv2.imshow("Skeleton", skeleton)

        except Exception as e:
            logger.error(f"Error processing frame: {e}")

    def draw_overlay(self, frame: np.ndarray):
        """Draw the current word, sentence, and suggestions on the frame."""
        # Current word
        cv2.putText(frame, f"Word: {self.current_word}", (10, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        # Sentence
        cv2.putText(frame, f"Sentence: {self.sentence}", (10, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Word suggestions
        if self.current_word:
            suggestions = self.spell.candidates(self.current_word)
            if suggestions:
                suggestion_text = "Suggestions: " + ", ".join(list(suggestions)[:3])
                cv2.putText(frame, suggestion_text, (10, 200),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    def run(self):
        """Main application loop."""
        logger.info("🎬 Starting gesture recognition with skeleton model...")
        logger.info("Press 'q' to exit")

        try:
            while True:
                success, frame = self.cap.read()

                if not success:
                    logger.warning("Failed to read frame from camera")
                    break

                self.process_frame(frame)
                self.draw_overlay(frame)

                cv2.imshow("ASL to Text (Skeleton Model)", frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    logger.info("Exit requested")
                    break

        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        except Exception as e:
            logger.error(f"Error in main loop: {e}")
        finally:
            self.cleanup()

    def cleanup(self):
        """Clean up resources."""
        try:
            if self.cap is not None:
                self.cap.release()
            cv2.destroyAllWindows()
            if self.hands is not None:
                self.hands.close()
            logger.info("Cleanup complete")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")


def main():
    """Entry point for the application."""
    config = Config()
    app = GestureRecognitionApp(config)

    if not app.validate_setup():
        return 1
    if not app.initialize():
        return 1

    app.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())