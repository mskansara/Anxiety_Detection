# def preprocess_image(self, image: np.array) -> np.array:
#     """
#     Preprocesses the image to detect and crop the face, and converts it to RGB.

#     Args:
#     - image (np.array): The original image.

#     Returns:
#     - np.array: The processed image ready for face detection and mood analysis.
#     """
#     try:
#         # Convert to grayscale for face detection
#         gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#         # Focus on the central part of the image
#         height, width = gray_image.shape
#         central_region = gray_image[
#             height // 4 : 3 * height // 4, width // 4 : 3 * width // 4
#         ]

#         # Adjust detection parameters for different conditions
#         faces = self.face_cascade.detectMultiScale(
#             central_region,
#             scaleFactor=1.1,
#             minNeighbors=5,
#             minSize=(30, 30),
#             flags=cv2.CASCADE_SCALE_IMAGE,
#         )

#         if len(faces) == 0:
#             return None  # No faces detected

#         # Adjust face coordinates to the original image dimensions
#         print(faces[0])
#         x, y, w, h = faces[0]
#         x += width // 4
#         y += height // 4

#         padding = 100  # Add padding to avoid too-tight cropping
#         x = max(0, x - padding)
#         y = max(0, y - padding)
#         w += 2 * padding
#         h += 2 * padding

#         # Draw a rectangle around the detected face for debugging
#         cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)

#         # Crop the face
#         face_image = image[y : y + h, x : x + w]

#         # Convert the face image to RGB
#         rgb_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)

#         # Resize to the required size for the model (e.g., 224x224 or 48x48)
#         resized_image = cv2.resize(rgb_image, (48, 48))

#         # Rotate the image 90 degrees to the right
#         rotated_image = cv2.rotate(resized_image, cv2.ROTATE_90_CLOCKWISE)

#         debug_filename = f"face_detected_{self.image_counter}.jpg"
#         cv2.imwrite(debug_filename, rotated_image)
#         print(f"Face detected and saved as {debug_filename}")

#         return rotated_image
#     except Exception as e:
#         print(f"Error in preprocessing image: {e}")
#         return None
