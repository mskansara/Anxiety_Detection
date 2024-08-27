import cv2
from fer import FER
import numpy as np


class ImageProcessor:
    def __init__(self) -> None:
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        self.image_counter = 0

    def detect_mood(self, image):
        emotion_detector = FER(mtcnn=True)
        image = np.rot90(image, -1)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        analysis = emotion_detector.detect_emotions(image)
        if analysis:
            emotion, score = emotion_detector.top_emotion(image)
            return emotion, score
        else:
            return None, None
