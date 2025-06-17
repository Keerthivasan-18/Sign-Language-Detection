import cv2
import mediapipe as mp
import numpy as np
import os
import imgaug.augmenters as iaa

# Set your directories
val_dir = 'asl_alphabet_test'
train_dir = 'asl_alphabet_train'

# Initialize MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=2,
    min_detection_confidence=0.05,
    min_tracking_confidence=0.5
)

# Optional: image augmentation for training
def augment_training_imgs(image):
    transform = iaa.Sequential([
        iaa.Affine(rotate=(-5, 5))  # Random rotation
    ], random_order=True)
    return transform.augment_image(image)

# Process a single image
def process_image(image_path, target_size=(200, 200), occupy_percent=0.8, is_training=False):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Failed to read image: {image_path}")
        return False

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)

    if not results.multi_hand_landmarks:
        print(f"No hands detected in: {image_path}")
        return False

    height, width, _ = image.shape
    hand_mask = np.zeros((height, width), dtype=np.uint8)

    for hand_landmarks in results.multi_hand_landmarks:
        points = [(int(lm.x * width), int(lm.y * height)) for lm in hand_landmarks.landmark]
        cv2.fillPoly(hand_mask, [np.array(points, dtype=np.int32)], 255)

    kernel = np.ones((5, 5), np.uint8)
    hand_mask_dilated = cv2.dilate(hand_mask, kernel, iterations=1)
    hand_mask_blurred = cv2.GaussianBlur(hand_mask_dilated, (21, 21), 0)

    hand_on_black = np.where(hand_mask_blurred[..., None] > 0, image, np.zeros_like(image))

    contours, _ = cv2.findContours(hand_mask_blurred, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return False

    x_min, y_min, x_max, y_max = width, height, 0, 0
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        x_min, y_min = min(x_min, x), min(y_min, y)
        x_max, y_max = max(x_max, x + w), max(y_max, y + h)

    hand_area = hand_on_black[y_min:y_max, x_min:x_max]
    hand_w, hand_h = x_max - x_min, y_max - y_min
    scale = min(target_size[0] / hand_w, target_size[1] / hand_h) * occupy_percent
    scaled_w, scaled_h = int(hand_w * scale), int(hand_h * scale)

    resized_hand = cv2.resize(hand_area, (scaled_w, scaled_h), interpolation=cv2.INTER_AREA)

    centered_image = np.zeros(target_size + (3,), dtype=np.uint8)
    x_offset = (target_size[0] - scaled_w) // 2
    y_offset = (target_size[1] - scaled_h) // 2
    centered_image[y_offset:y_offset + scaled_h, x_offset:x_offset + scaled_w] = resized_hand

    if is_training:
        centered_image = augment_training_imgs(centered_image)

    gray_img = cv2.cvtColor(centered_image, cv2.COLOR_RGB2GRAY)
    base, ext = os.path.splitext(image_path)
    new_path = f"{base}_mediapipe{ext}"

    if cv2.imwrite(new_path, gray_img):
        os.remove(image_path)
        print(f"Processed and saved: {new_path}")
        return True
    else:
        return False

# Process all images in a directory
def process_directory(directory):
    deleted_count = {}
    total_processed = total_remaining = total_deleted = 0

    for subdir, dirs, files in os.walk(directory):
        subdir_name = os.path.basename(subdir)
        num_deleted = processed = 0
        total_files = len(files)
        print(f"\nProcessing directory: {subdir_name}")

        is_training = directory == train_dir
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                file_path = os.path.join(subdir, file)
                if not process_image(file_path, is_training=is_training):
                    os.remove(file_path)
                    num_deleted += 1
                else:
                    total_remaining += 1
                processed += 1
                print(f"Progress: {processed}/{total_files}")

        deleted_count[subdir_name] = num_deleted
        total_processed += total_files
        total_deleted += num_deleted

    return deleted_count, total_remaining, total_processed, total_deleted

# Run processing
if __name__ == "__main__":
    for directory in [train_dir, val_dir]:
        deleted, remaining, processed, deleted_total = process_directory(directory)
        print(f"\n--- Results for {directory} ---")
        for cls, cnt in deleted.items():
            print(f"{cls}: {cnt} images deleted")
        print(f"Total processed: {processed}")
        print(f"Total deleted: {deleted_total}")
        print(f"Total remaining: {remaining}")

    hands.close()
