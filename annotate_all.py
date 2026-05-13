import cv2
import mediapipe as mp
import os
import sys
from tqdm import tqdm

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)

def annotate_images(source_dir='subset_5000', target_dir='annotated_subset'):
    """
    Process all images in source_dir, draw landmarks if hand detected,
    and save annotated images to target_dir (mirroring folder structure).
    Returns counts of total, success, failure.
    """
    # Create target directory if it doesn't exist
    os.makedirs(target_dir, exist_ok=True)

    total = 0
    success = 0
    failure = 0

    # Walk through all class folders in source_dir
    for class_name in os.listdir(source_dir):
        class_path = os.path.join(source_dir, class_name)
        if not os.path.isdir(class_path):
            continue

        # Create corresponding class folder in target_dir
        target_class_path = os.path.join(target_dir, class_name)
        os.makedirs(target_class_path, exist_ok=True)

        # Get all image files in this class folder
        image_files = [f for f in os.listdir(class_path) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
        
        for img_file in tqdm(image_files, desc=f"Processing {class_name}"):
            img_path = os.path.join(class_path, img_file)
            total += 1

            try:
                image = cv2.imread(img_path)
                if image is None:
                    print(f"Warning: Could not read {img_path}")
                    failure += 1
                    continue

                # Convert to RGB for MediaPipe
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = hands.process(image_rgb)

                if results.multi_hand_landmarks:
                    # Draw landmarks on the image
                    for hand_landmarks in results.multi_hand_landmarks:
                        mp_draw.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    
                    # Save annotated image
                    target_img_path = os.path.join(target_class_path, img_file)
                    cv2.imwrite(target_img_path, image)
                    success += 1
                else:
                    failure += 1
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
                failure += 1

    return total, success, failure

if __name__ == "__main__":
    total, success, failure = annotate_images()
    print("\n--- Annotation Summary ---")
    print(f"Total images processed: {total}")
    print(f"Successfully annotated (hand detected): {success}")
    print(f"Failed (no hand detected or error): {failure}")
    if total > 0:
        print(f"Success rate: {success/total*100:.2f}%")
    
    print("\nPossible reasons for failed annotations:")
    print("- Hand not clearly visible (occluded, too far, blurry)")
    print("- Poor lighting conditions")
    print("- Image contains no hand (but dataset should have hand signs)")
    print("- MediaPipe confidence threshold not met (default 0.5)")
    print("- Image format issues or corruption")
    print("- Multiple hands present (max_num_hands=1 may ignore some)")