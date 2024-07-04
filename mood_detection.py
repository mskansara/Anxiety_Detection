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
    def __init__(self):
        self.rgb_image = None
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        self.image_counter = 0

    def on_image_received(self, image: np.array, record: ImageDataRecord):
        self.rgb_image = image

    def detect_mood(self):
        """
        Detects mood from images in the queue using DeepFace with a local model.
        """
        emotion_detector = FER(mtcnn=True)
        while True:
            try:
                image = self.rgb_image
                image = np.rot90(image, -1)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                # print(image)
                if image is None:  # Termination signal
                    break

                analysis = emotion_detector.detect_emotions(image)
                emotion, score = emotion_detector.top_emotion(image)

                if analysis:
                    print(emotion, score)
                    continue

            except Exception as e:
                print(f"No face detected")
                continue


# Main function to set up and manage streaming
def main():
    args = parse_args()
    project_aria_handler = ProjectARIAHandler(args)
    [streaming_client, streaming_manager, device_client, device] = (
        project_aria_handler.setup_project_aria()
    )
    try:
        streaming_manager.start_streaming()
        print(f"Streaming state: {streaming_manager.streaming_state}")
        observer = StreamingClientObserver()
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
