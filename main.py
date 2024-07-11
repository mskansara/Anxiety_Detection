import numpy as np
import argparse
import sys
import aria.sdk as aria
import time
from queue import Queue
import threading
from faster_whisper import WhisperModel
from projectaria_tools.core.sensor_data import (
    AudioData,
    AudioDataRecord,
    ImageData,
    ImageDataRecord,
)
import cv2
from deepface import DeepFace
from fer import FER
import os
from dotenv import load_dotenv
import torch

# Load environment variables
load_dotenv()
AUTH_TOKEN = os.getenv("AUTH_TOKEN")

# Constants
SAMPLE_RATE = 48000
TARGET_SAMPLE_RATE = 16000
NUM_CHANNELS = 7
BUFFER_DURATION = 10
RGB_SUBSCRIBER_DATA_TYPE = aria.StreamingDataType.Rgb
AUDIO_SUBSCRIBER_DATA_TYPE = aria.StreamingDataType.Audio

colour_codes = {"neutral": "white", "happy": "blue", "sad": "orange", "angry": "red"}


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
        elif self.args.streaming_interface == "wifi":
            streaming_config.streaming_interface = aria.StreamingInterface.WifiStation

        streaming_config.security_options.use_ephemeral_certs = True
        self.streaming_manager.streaming_config = streaming_config

        config = self.streaming_client.subscription_config
        config.subscriber_data_type = (
            aria.StreamingDataType.Rgb | aria.StreamingDataType.Audio
        )
        config.message_queue_size[AUDIO_SUBSCRIBER_DATA_TYPE] = 100
        config.message_queue_size[RGB_SUBSCRIBER_DATA_TYPE] = 100
        config.security_options.use_ephemeral_certs = True
        self.streaming_client.subscription_config = config

        return [
            self.streaming_client,
            self.streaming_manager,
            self.device_client,
            device,
        ]

    def update_iptables(self):
        pass


class AudioProcessor:
    def __init__(self):
        self.lock = threading.Lock()

    def process_multi_channel_audio(self, audio_data, num_channels):
        total_samples = len(audio_data)
        if total_samples % num_channels != 0:
            truncated_samples = total_samples - (total_samples % num_channels)
            audio_data = audio_data[:truncated_samples]

        audio_data = audio_data.reshape(-1, num_channels)
        mono_audio = np.mean(audio_data, axis=1)
        return mono_audio

    def resample_audio(self, audio_data, original_sample_rate, target_sample_rate):
        num_samples = int(len(audio_data) * target_sample_rate / original_sample_rate)
        resampled_data = np.interp(
            np.linspace(0, len(audio_data), num_samples),
            np.arange(len(audio_data)),
            audio_data,
        )
        return resampled_data.astype(np.float32)

    def normalize_audio(self, audio_data):
        max_val = np.max(np.abs(audio_data))
        return audio_data / max_val if max_val > 0 else audio_data

    def transcribe_and_diarize_audio(self, model, audio_queue, output_folder):
        target_chunk_size = int(TARGET_SAMPLE_RATE * BUFFER_DURATION)
        while True:
            try:
                audio_data = []
                while len(audio_data) < target_chunk_size:
                    self.lock.acquire()
                    try:
                        # Extract the audio data and other information from the dictionary
                        queue_item = audio_queue.get(timeout=1)
                        chunk_data = queue_item.get["audio_data"]
                        timestamp = queue_item.get("timestamp", None)
                        needed_samples = target_chunk_size - len(audio_data)
                        if not audio_queue.empty():
                            queue_data = audio_queue.get(timeout=1)
                            chunk_data = queue_data[:needed_samples]
                            audio_data.extend(chunk_data)
                            if len(queue_data) > needed_samples:
                                remaining_data = queue_data[needed_samples:]
                                audio_queue.put(
                                    {
                                        "audio_data": remaining_data,
                                        "timestamp": timestamp,
                                    }
                                )
                    finally:
                        self.lock.release()
                    time.sleep(0.1)

                if audio_data:
                    audio_data = np.array(audio_data, dtype=np.float32)
                    segments, _ = model.transcribe(
                        audio_data, beam_size=5, language="en"
                    )
                    transcription = "\n".join(segment.text for segment in segments)
                    print(str(timestamp) + " - " + transcription)
                    waveform = torch.tensor([audio_data], dtype=torch.float32)
                    print(waveform)
            except Exception as e:
                print(f"Error during transcription: {e}")
            finally:
                time.sleep(0.1)


class StreamingClientObserver:
    def __init__(self, audio_queue, audio_processor):
        self.audio_timestamp = None
        self.audio_queue = audio_queue
        self.audio_buffer = np.array([], dtype=np.float32)
        self.sample_rate = SAMPLE_RATE
        self.lock = threading.Lock()
        self.audio_processor = audio_processor
        self.rgb_image = {"data": None, "timestamp": None}
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        self.image_counter = 0

    def on_audio_received(self, audio_data: AudioData, record: AudioDataRecord):
        self.audio_timestamp = record.capture_timestamps_ns
        print(record.capture_timestamps_ns)
        exit()
        audio_data_values = np.array(audio_data.data, dtype=np.float32)
        num_samples_per_channel = len(audio_data_values) // NUM_CHANNELS
        audio_data_values = audio_data_values[: num_samples_per_channel * NUM_CHANNELS]
        mono_audio = self.audio_processor.process_multi_channel_audio(
            audio_data_values, NUM_CHANNELS
        )
        self.audio_buffer = np.concatenate((self.audio_buffer, mono_audio))
        # print(self.audio_buffer)
        buffer_sample_count = int(SAMPLE_RATE * BUFFER_DURATION)
        if len(self.audio_buffer) >= buffer_sample_count:
            buffer_audio = self.audio_buffer[:buffer_sample_count]
            self.audio_buffer = self.audio_buffer[buffer_sample_count:]

            resampled_audio = self.audio_processor.resample_audio(
                buffer_audio, SAMPLE_RATE, TARGET_SAMPLE_RATE
            )
            normalized_audio = self.audio_processor.normalize_audio(resampled_audio)

            self.lock.acquire()
            # self.audio_queue.put(normalized_audio.tolist())
            self.audio_queue.put(
                {
                    "audio_data": normalized_audio.tolist(),
                    "timestamp": record.capture_timestamp_ns,
                }
            )
            self.lock.release()

    def on_image_received(self, image: np.array, record: ImageDataRecord):
        self.rgb_image["data"] = image
        self.rgb_image["timestamp"] = record.capture_timestamp_ns

    def detect_mood(self):
        emotion_detector = FER(mtcnn=True)
        while True:
            try:
                image = self.rgb_image["data"]
                image = np.rot90(image, -1)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                if image is None:
                    break
                analysis = emotion_detector.detect_emotions(image)
                emotion, score = emotion_detector.top_emotion(image)
                if analysis:
                    mood = {
                        "timestamp": self.rgb_image["timestamp"],
                        "emotion": emotion,
                        "score": score,
                        "colour": colour_codes[emotion],
                    }
                    print(mood)
                    continue
            except Exception as e:
                print(f"No face detected: {e}")
                continue


def main():
    output_folder = "./transcriptions"
    args = parse_args()
    project_aria_handler = ProjectARIAHandler(args)
    [streaming_client, streaming_manager, device_client, device] = (
        project_aria_handler.setup_project_aria()
    )

    try:
        streaming_manager.start_streaming()
        print(f"Streaming state: {streaming_manager.streaming_state}")

        audio_queue = Queue()
        audio_processor = AudioProcessor()
        observer = StreamingClientObserver(audio_queue, audio_processor)
        streaming_client.set_streaming_client_observer(observer)

        print("Start listening to audio and image data")
        streaming_client.subscribe()

        model = WhisperModel("tiny.en", device="cpu", compute_type="float32")
        transcription_thread = threading.Thread(
            target=audio_processor.transcribe_and_diarize_audio,
            args=(model, audio_queue, output_folder),
            daemon=True,
        )
        transcription_thread.start()
        print("Transcription thread started")

        detection_thread = threading.Thread(target=observer.detect_mood, daemon=True)
        detection_thread.start()
        print("Mood detection thread started")

        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Exiting...")
    finally:
        print("Stop listening to audio and image data")
        streaming_client.unsubscribe()
        streaming_manager.stop_streaming()
        device_client.disconnect(device)


if __name__ == "__main__":
    main()
