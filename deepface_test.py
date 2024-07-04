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
    img_path="./images/Happy.jpg",
    detector_backend=backends[5],
    align=alignment_modes[0],
)

for face in face_objs:
    detected_face = face["face"]

    resized_face = (detected_face * 255).astype(np.int8)
    print(resized_face)
    analysis = DeepFace.analyze(img_path=resized_face, enforce_detection=False)


print(analysis)

# face_objs = DeepFace.extract_faces(
#     img_path=image,
#     detector_backend=backends[4],
#     align=alignment_modes[0],
# )

# if face_objs is None:
#     print("No face detected in the image.")
#     continue

# # Use DeepFace to analyze the image

# for face_obj in face_objs:
#     # print(face_obj)
#     print("Face Detected")
#     face_image = face_obj["face"]
#     # Convert the face image to a format DeepFace expects
#     if face_image.dtype != np.uint8:
#         face_image = (255.0 * face_image).astype(np.uint8)
#         # print(face_image)
#     analysis = DeepFace.analyze(
#         img_path=face_image,
#         actions=["emotion"],
#         enforce_detection=False,
#     )
#     print(analysis)
#     # Extract the dominant emotion and its confidence
#     dominant_emotion = analysis["dominant_emotion"]
#     confidence = analysis["emotion"][dominant_emotion]

#     print(
#         f"Detected mood: {dominant_emotion} with confidence: {confidence:.2f}%"
#     )
