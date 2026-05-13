import cv2
import mediapipe as mp
import os
import argparse
from tqdm import tqdm

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)

def annotate_and_save(input_dir, output_dir):
    """
    Process all images in input_dir (with class subfolders), annotate with hand landmarks,
    and save to output_dir mirroring structure. Count successes.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    total = 0
    success = 0
    failed = 0
    
    # Get all class folders
    classes = [d for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))]
    
    for cls in classes:
        cls_input_path = os.path.join(input_dir, cls)
        cls_output_path = os.path.join(output_dir, cls)
        os.makedirs(cls_output_path, exist_ok=True)
        
        # Get image files in this class
        image_files = [f for f in os.listdir(cls_input_path) 
                       if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
        total += len(image_files)
        
        for img_file in tqdm(image_files, desc=f"Processing {cls}"):
            img_path = os.path.join(cls_input_path, img_file)
            image = cv2.imread(img_path)
            if image is None:
                print(f"Warning: Could not read {img_path}, skipping.")
                failed += 1
                continue
            
            # Convert to RGB for MediaPipe
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = hands.process(image_rgb)
            
            if results.multi_hand_landmarks:
                # Draw landmarks on original image (BGR)
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_draw.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                # Save annotated image
                out_path = os.path.join(cls_output_path, img_file)
                cv2.imwrite(out_path, image)
                success += 1
            else:
                failed += 1
                # Optionally save original without annotations? Not requested.
                # We'll just count as failure.
    
    return total, success, failed

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Annotate all images with MediaPipe hand landmarks and save to separate folder.")
    parser.add_argument("--input", type=str, default="subset_5000", help="Input directory containing class subfolders with images.")
    parser.add_argument("--output", type=str, default="annotated_images", help="Output directory to save annotated images.")
    args = parser.parse_args()
    
    total, success, failed = annotate_and_save(args.input, args.output)
    
    print("\n=== Summary ===")
    print(f"Total images processed: {total}")
    print(f"Successfully annotated (hand detected): {success}")
    print(f"Failed (no hand detected or error): {failed}")
    if total > 0:
        print(f"Success rate: {success/total*100:.2f}%")