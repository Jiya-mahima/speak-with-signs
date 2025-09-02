import cv2
import os

# Define paths
input_folder = "dataset"  # Change to your dataset folder path
output_folder = "dataset_resized"  # New folder for resized images
os.makedirs(output_folder, exist_ok=True)

# Resize settings
IMG_SIZE = (48, 48)  # Model input size

# Loop through all images in the dataset folder
for root, _, files in os.walk(input_folder):
    for file in files:
        if file.endswith(('.png', '.jpg', '.jpeg')):  # Filter image files
            input_path = os.path.join(root, file)
            output_path = os.path.join(output_folder, file)

            # Read image
            img = cv2.imread(input_path)
            if img is None:
                print(f"❌ Skipping {file}: Unable to read")
                continue

            # Resize image
            img_resized = cv2.resize(img, IMG_SIZE)

            # Save resized image
            cv2.imwrite(output_path, img_resized)

            print(f"✅ Resized & Saved: {output_path}")

print("🎯 All images resized successfully!")
