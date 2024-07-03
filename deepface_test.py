from deepface import DeepFace
import numpy as np

backends = [
    "opencv",
    "ssd",
    "dlib",
    "mtcnn",
    "fastmtcnn",
    "retinaface",
    "mediapipe",
    "yolov8",
    "yunet",
    "centerface",
]

alignment_modes = [True, False]

# Face detection and alignment
face_objs = DeepFace.extract_faces(
    img_path="/images/Happy.jpg", detector_backend=backends[4], align=alignment_modes[0]
)

for face in face_objs:
    detected_face = face["face"]

    resized_face = (detected_face * 255).astype(np.int8)
    print(resized_face)
    analysis = DeepFace.analyze(img_path=resized_face, enforce_detection=False)


print(analysis)
