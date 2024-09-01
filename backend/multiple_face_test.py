import unittest
from ImageClass import ImageProcessor
import cv2


class TestImageProcessorMultipleFaces(unittest.TestCase):

    def setUp(self):
        self.image_processor = ImageProcessor()

    def test_multiple_faces(self):
        # Load a test image with multiple faces
        image = cv2.imread("../images/Face_Images/multiple_faces_1.png")

        # Verify that the image is loaded correctly
        if image is None:
            self.fail(
                "Failed to load the test image. Check the image path and try again."
            )

        # Detect moods in the image
        detected_emotions = self.image_processor.detect_mood(image)

        # Check if emotions were detected
        self.assertTrue(
            len(detected_emotions) > 0, "No emotions detected in the image."
        )

        for i, (emotion, score) in enumerate(detected_emotions):
            print(f"Face {i+1}: Detected emotion: {emotion}, Score: {score}")


if __name__ == "__main__":
    unittest.main()
