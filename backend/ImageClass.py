import cv2
from fer import FER
import numpy as np


class ImageProcessor:
    def __init__(self) -> None:
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

    def detect_mood(self, image):
        if not isinstance(image, np.ndarray):
            raise ValueError("Invalid image provided. Image must be a numpy array.")

        emotion_detector = FER(mtcnn=True)
        # image = np.rot90(image, -1)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        faces = self.face_cascade.detectMultiScale(
            image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )

        print(f"Number of faces detected: {len(faces)}")
        detected_emotions = []

        for x, y, w, h in faces:
            roi = image[y : y + h, x : x + w]
            print(f"Original face region: {(x, y, w, h)}")
            cv2.imshow("Original Face ROI", roi)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

            # Resize the face region to a larger size
            roi_resized = cv2.resize(roi, (128, 128))
            cv2.imshow("Resized Face ROI", roi_resized)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

            analysis = emotion_detector.detect_emotions(roi_resized)
            print(f"Analysis for face at {(x, y, w, h)}: {analysis}")

            if analysis:
                top_emotion = analysis[0]["emotions"]
                top_emotion = sorted(
                    top_emotion.items(), key=lambda item: item[1], reverse=True
                )[0]
                detected_emotions.append(top_emotion)

        return detected_emotions
