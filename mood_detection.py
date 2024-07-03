import numpy as np
import argparse
import sys
import aria.sdk as aria
import time
from queue import Queue, Empty
import threading
from projectaria_tools.core.sensor_data import ImageData, ImageDataRecord
import cv2
from deepface import DeepFace
import json
from fer import FER

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


# Argument parsing
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--interface",
        dest="streaming_interface",
        type=str,
        required=True,
        help="Type of interface to use for streaming. Options are usb or wifi.",
        choices=["usb", "wifi"],
    )
    parser.add_argument(
        "--update_iptables",
        default=False,
        action="store_true",
        help="Update iptables to enable receiving the data stream, only for Linux.",
    )
    parser.add_argument(
        "--profile",
        dest="profile_name",
        type=str,
        default="profile18",
        required=False,
        help="Profile to be used for streaming.",
    )
    parser.add_argument(
        "--device-ip", help="IP address to connect to the device over wifi"
    )
    return parser.parse_args()


class ProjectARIAHandler:
    def __init__(self, args):
        self.args = args
        self.device_client = None
        self.streaming_manager = None
        self.streaming_client = None
        self.audio_queue = None

    def setup_project_aria(self):
        if self.args.update_iptables and sys.platform.startswith("linux"):
            self.update_iptables()

        aria.set_log_level(aria.Level.Info)
        self.device_client = aria.DeviceClient()
        client_config = aria.DeviceClientConfig()
        if self.args.device_ip:
            client_config.ip_v4_address = self.args.device_ip
        self.device_client.set_client_config(client_config)
        device = self.device_client.connect()

        self.streaming_manager = device.streaming_manager
        self.streaming_client = self.streaming_manager.streaming_client

        streaming_config = aria.StreamingConfig()
        streaming_config.profile_name = self.args.profile_name
        if self.args.streaming_interface == "usb":
            streaming_config.streaming_interface = aria.StreamingInterface.Usb
        if self.args.streaming_interface == "wifi":
            streaming_config.streaming_interface = aria.StreamingInterface.WifiStation
        streaming_config.security_options.use_ephemeral_certs = True
        self.streaming_manager.streaming_config = streaming_config
        config = self.streaming_client.subscription_config
        config.subscriber_data_type = aria.StreamingDataType.Rgb
        config.message_queue_size[aria.StreamingDataType.Rgb] = 100
        config.security_options.use_ephemeral_certs = True
        self.streaming_client.subscription_config = config

        return [
            self.streaming_client,
            self.streaming_manager,
            self.device_client,
            device,
        ]

    def update_iptables(self):
        # Implement iptables update logic here if needed for Linux
        pass


class StreamingClientObserver:
    def __init__(self, image_queue: Queue):
        self.rgb_image = None
        self.image_queue = image_queue
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        self.image_counter = 0

    def on_image_received(self, image: np.array, record: ImageDataRecord):
        self.image_queue.put(image)

    def detect_mood(self):
        """
        Detects mood from images in the queue using DeepFace with a local model.
        """
        emotion_detector = FER(mtcnn=True)
        while True:
            try:
                image = self.image_queue.get(timeout=1)  # Wait for 1 second
                print(image)
                if image is None:  # Termination signal
                    break

                analysis = emotion_detector.detect_emotions(image)

                if analysis:
                    print(analysis)
                    continue

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
                #         print(face_image)
                #     analysis = DeepFace.analyze(
                #         img_path=face_image,
                #         actions=["emotion"],
                #         enforce_detection=False
                #     )
                #     print(analysis)
                #     # Extract the dominant emotion and its confidence
                #     dominant_emotion = analysis["dominant_emotion"]
                #     confidence = analysis["emotion"][dominant_emotion]

                #     print(
                #         f"Detected mood: {dominant_emotion} with confidence: {confidence:.2f}%"
                #     )

            except Empty:
                continue  # No image in the queue, continue waiting
            except Exception as e:
                # print(f"Error detecting mood: {e}")
                continue

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


# Main function to set up and manage streaming
def main():
    args = parse_args()
    project_aria_handler = ProjectARIAHandler(args)
    [streaming_client, streaming_manager, device_client, device] = (
        project_aria_handler.setup_project_aria()
    )
    image_queue = Queue()
    try:
        streaming_manager.start_streaming()
        print(f"Streaming state: {streaming_manager.streaming_state}")
        observer = StreamingClientObserver(image_queue)
        streaming_client.set_streaming_client_observer(observer)

        print("Start listening to image data")
        streaming_client.subscribe()
        detection_thread = threading.Thread(target=observer.detect_mood, daemon=True)
        detection_thread.start()
        print("Mood detection thread started")
        while True:

            time.sleep(1)
    except KeyboardInterrupt:
        print("Exiting...")
    finally:
        print("Stop listening to image data")
        streaming_client.unsubscribe()
        streaming_manager.stop_streaming()
        device_client.disconnect(device)


if __name__ == "__main__":
    main()
