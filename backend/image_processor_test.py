import os
import cv2
import numpy as np
from ImageClass import ImageProcessor

# Directory containing the image folders
image_dir = "../images/Face_Images/images/images/validation"

# Initialize the image processor
image_processor = ImageProcessor()

# Prepare the dataset
dataset = []
for emotion_dir in os.listdir(image_dir):
    emotion_path = os.path.join(image_dir, emotion_dir)
    if os.path.isdir(emotion_path):
        image_count = 0  # Initialize counter for images in each directory
        for image_file in os.listdir(emotion_path):
            if image_file.endswith((".jpg")):  # Filter image files
                image_path = os.path.join(emotion_path, image_file)
                dataset.append((image_path, emotion_dir))
                image_count += 1
                if image_count >= 500:  # Stop after 10 images
                    break

# Initialize counters for accuracy calculation
correct_predictions = 0
valid_predictions = 0  # Counter for valid predictions

total_predictions = len(dataset)

# Process each image and calculate accuracy
for image_path, ground_truth_emotion in dataset:
    image = cv2.imread(image_path)
    predicted_emotion, _ = image_processor.detect_mood(image)

    # Only count predictions that are not None
    if predicted_emotion is not None:
        valid_predictions += 1
        if predicted_emotion == ground_truth_emotion:
            correct_predictions += 1

# Calculate accuracy based on valid predictions
if valid_predictions > 0:
    accuracy = correct_predictions / valid_predictions * 100
    print(f"Model Accuracy: {accuracy:.2f}%")
else:
    print("No valid predictions were made.")
print(correct_predictions)
