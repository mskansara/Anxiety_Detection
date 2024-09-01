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
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
import asyncio
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import json


from backend.database_connection import DatabaseConnection

# Load environment variables
load_dotenv()
AUTH_TOKEN = os.getenv("AUTH_TOKEN")
MONGODB_AUTH = os.getenv("MONGODB_AUTH")
uri = f"mongodb+srv://manthankansara7:{MONGODB_AUTH}@cluster0.4ubii7g.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"


# Constants
SAMPLE_RATE = 48000
TARGET_SAMPLE_RATE = 16000
NUM_CHANNELS = 7
BUFFER_DURATION = 10  # This needs to be tested
RGB_SUBSCRIBER_DATA_TYPE = aria.StreamingDataType.Rgb
AUDIO_SUBSCRIBER_DATA_TYPE = aria.StreamingDataType.Audio

colour_codes = {
    "neutral": "black",
    "happy": "blue",
    "sad": "orange",
    "angry": "red",
    "fear": "green",
}

# Global variables to hold transcriptions and detected moods
transcriptions = []
detected_mood = []

app = FastAPI()

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


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


dummy_data = {
    "transcriptions": [
        {
            "start_timestamp": "940710969012",
            "end_timestamp": "941510981137",
            "transcription": "Patient: I feel anxious sometimes. Doctor: Can you tell me more about that?",
        },
        {
            "start_timestamp": "941520981138",
            "end_timestamp": "942310971050",
            "transcription": "Patient: It's like there's always this feeling of unease, especially in social situations. Doctor: When did you first notice these feelings?",
        },
        {
            "start_timestamp": "942320971051",
            "end_timestamp": "943010971550",
            "transcription": "Patient: I think it started in high school. I used to get really nervous before class presentations. Doctor: Have these feelings of anxiety gotten better or worse over time?",
        },
        {
            "start_timestamp": "943020971551",
            "end_timestamp": "943810968887",
            "transcription": "Patient: They have gotten worse. Now, even going to the grocery store makes me anxious. Doctor: That sounds challenging. Have you found any strategies that help manage your anxiety?",
        },
        {
            "start_timestamp": "943820968888",
            "end_timestamp": "944510969387",
            "transcription": "Patient: Sometimes deep breathing helps, but not always. Doctor: It's good that you have a coping mechanism. Have you tried any other techniques or therapies?",
        },
        {
            "start_timestamp": "944520969388",
            "end_timestamp": "945810973062",
            "transcription": "Patient: I tried meditation, but I find it hard to focus. Doctor: Meditation can be difficult at first. It might help to start with shorter sessions and gradually increase the time.",
        },
        {
            "start_timestamp": "945820973063",
            "end_timestamp": "946710972300",
            "transcription": "Patient: I will try that. Doctor: Do you have a support system you can rely on, like friends or family?",
        },
        {
            "start_timestamp": "946720972301",
            "end_timestamp": "947710972425",
            "transcription": "Patient: My family is supportive, but they don't really understand what I'm going through. Doctor: It can be hard for others to understand. Have you considered joining a support group?",
        },
        {
            "start_timestamp": "947720972426",
            "end_timestamp": "948710974850",
            "transcription": "Patient: I haven't, but maybe I should. Doctor: Support groups can provide a sense of community and understanding. It might be worth looking into.",
        },
        {
            "start_timestamp": "948720974851",
            "end_timestamp": "949610976512",
            "transcription": "Patient: I'll think about it. Doctor: That's good to hear. Remember, you're not alone in this, and there are people who can help.",
        },
        {
            "start_timestamp": "949620976513",
            "end_timestamp": "950610966887",
            "transcription": "Patient: Thank you, doctor. Doctor: You're welcome. Let's continue to work on this together.",
        },
    ],
    "detected_mood": [
        {
            "timestamp": "940710969012",
            "emotion": "neutral",
            "score": 0.74,
            "colour": "white",
        },
        {
            "timestamp": "941520981138",
            "emotion": "anxious",
            "score": 0.65,
            "colour": "orange",
        },
        {
            "timestamp": "942320971051",
            "emotion": "nervous",
            "score": 0.72,
            "colour": "yellow",
        },
        {
            "timestamp": "943020971551",
            "emotion": "anxious",
            "score": 0.77,
            "colour": "orange",
        },
        {
            "timestamp": "943820968888",
            "emotion": "frustrated",
            "score": 0.68,
            "colour": "red",
        },
        {
            "timestamp": "944520969388",
            "emotion": "neutral",
            "score": 0.70,
            "colour": "white",
        },
        {
            "timestamp": "945820973063",
            "emotion": "hopeful",
            "score": 0.66,
            "colour": "blue",
        },
        {
            "timestamp": "946720972301",
            "emotion": "neutral",
            "score": 0.67,
            "colour": "white",
        },
        {
            "timestamp": "947720972426",
            "emotion": "considerate",
            "score": 0.64,
            "colour": "green",
        },
        {
            "timestamp": "948720974851",
            "emotion": "thoughtful",
            "score": 0.71,
            "colour": "blue",
        },
        {
            "timestamp": "949620976513",
            "emotion": "grateful",
            "score": 0.78,
            "colour": "blue",
        },
        {
            "timestamp": "950610966887",
            "emotion": "relieved",
            "score": 0.81,
            "colour": "blue",
        },
    ],
}


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
        device_ip = None
        if device_ip:
            client_config.ip_v4_address = device_ip
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
        # print("Processing multi channel audio")
        total_samples = len(audio_data)
        if total_samples % num_channels != 0:
            truncated_samples = total_samples - (total_samples % num_channels)
            audio_data = audio_data[:truncated_samples]

        audio_data = audio_data.reshape(-1, num_channels)
        mono_audio = np.mean(audio_data, axis=1)
        return mono_audio

    def resample_audio(self, audio_data, original_sample_rate, target_sample_rate):
        # print("Resampling audio")
        num_samples = int(len(audio_data) * target_sample_rate / original_sample_rate)
        resampled_data = np.interp(
            np.linspace(0, len(audio_data), num_samples),
            np.arange(len(audio_data)),
            audio_data,
        )
        return resampled_data.astype(np.float32)

    def normalize_audio(self, audio_data):
        # print("Normalizing audio")
        max_val = np.max(np.abs(audio_data))
        return audio_data / max_val if max_val > 0 else audio_data

    def transcribe_and_diarize_audio(self, model, audio_queue, websocket):
        print("Transcribing audio")
        target_chunk_size = int(TARGET_SAMPLE_RATE * BUFFER_DURATION)
        while True:
            try:
                audio_data = []
                start_audio_timestamp = None
                end_audio_timestamp = None
                while len(audio_data) < target_chunk_size:
                    self.lock.acquire()
                    try:
                        if not audio_queue.empty():
                            queue_item = audio_queue.get(timeout=1)
                            chunk_data = queue_item["audio_data"]
                            start_audio_timestamp = queue_item.get(
                                "start_timestamp", start_audio_timestamp
                            )
                            end_audio_timestamp = queue_item.get(
                                "end_timestamp", end_audio_timestamp
                            )
                            needed_samples = target_chunk_size - len(audio_data)
                            audio_data.extend(chunk_data[:needed_samples])
                            if len(chunk_data) > needed_samples:
                                remaining_data = chunk_data[needed_samples:]
                                audio_queue.put(
                                    {
                                        "audio_data": remaining_data,
                                        "start_timestamp": start_audio_timestamp,
                                        "end_timestamp": end_audio_timestamp,
                                    }
                                )
                    finally:
                        self.lock.release()
                    time.sleep(0.1)

                if audio_data:
                    audio_data = np.array(audio_data, dtype=np.float32)
                    segments, _ = model.transcribe(
                        audio_data, beam_size=5, language="en", vad_filter=True
                    )
                    transcription = "\n".join(segment.text for segment in segments)

                    transcription_data = {
                        "transcription": transcription,
                        "start_timestamp": start_audio_timestamp,
                        "end_timestamp": end_audio_timestamp,
                    }
                    transcriptions.append(transcription_data)

                    data = {
                        "transcription": transcription,
                        "start_timestamp": start_audio_timestamp,
                        "end_timestamp": end_audio_timestamp,
                        "mood": None,
                    }
                    print(data)
                    asyncio.run(websocket.send_text(json.dumps(data)))
                    # waveform = torch.tensor([audio_data], dtype=torch.float32)
            except Exception as e:
                print(f"Error during transcription: {e}")
            finally:
                time.sleep(0.1)


class StreamingClientObserver:
    def __init__(self, audio_queue, audio_processor, websocket):
        self.start_audio_timestamp = None
        self.end_audio_timestamp = None
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
        self.websocket = websocket

    def on_audio_received(self, audio_data: AudioData, record: AudioDataRecord):

        self.start_audio_timestamp = record.capture_timestamps_ns[0]
        self.end_audio_timestamp = record.capture_timestamps_ns[
            len(record.capture_timestamps_ns) - 1
        ]

        audio_data_values = np.array(audio_data.data, dtype=np.float32)
        num_samples_per_channel = len(audio_data_values) // NUM_CHANNELS
        audio_data_values = audio_data_values[: num_samples_per_channel * NUM_CHANNELS]
        mono_audio = self.audio_processor.process_multi_channel_audio(
            audio_data_values, NUM_CHANNELS
        )
        self.audio_buffer = np.concatenate((self.audio_buffer, mono_audio))
        buffer_sample_count = int(SAMPLE_RATE * BUFFER_DURATION)
        if len(self.audio_buffer) >= buffer_sample_count:
            buffer_audio = self.audio_buffer[:buffer_sample_count]
            self.audio_buffer = self.audio_buffer[buffer_sample_count:]

            resampled_audio = self.audio_processor.resample_audio(
                buffer_audio, SAMPLE_RATE, TARGET_SAMPLE_RATE
            )
            normalized_audio = self.audio_processor.normalize_audio(resampled_audio)

            self.lock.acquire()
            self.audio_queue.put(
                {
                    "start_timestamp": self.start_audio_timestamp,
                    "end_timestamp": self.end_audio_timestamp,
                    "audio_data": normalized_audio.tolist(),
                }
            )
            self.lock.release()

    def on_image_received(self, image: np.array, record: ImageDataRecord):
        self.rgb_image["data"] = image
        self.rgb_image["timestamp"] = record.capture_timestamp_ns

    def detect_mood(self, detected_mood, websocket):
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
                    detected_mood.append(mood)
                    data = {
                        "transcription": None,
                        "start_timestamp": self.rgb_image["timestamp"],
                        "end_timestamp": self.rgb_image["timestamp"],
                        "mood": mood,
                    }
                    print(data)
                    asyncio.run(send_text_safe(websocket, data))
                    continue
            except Exception as e:
                # print(f"No face detected: {e}")
                continue


# Initialize global variables for threading and streaming
audio_queue = Queue()
audio_processor = AudioProcessor()
observer = None
project_aria_handler = None
transcription_thread = None
detection_thread = None
streaming_client = None
streaming_manager = None
device_client = None
device = None


class StartStreamingRequest(BaseModel):
    interface: str
    profile_name: str = "profile18"


@app.post("/start_streaming")
async def start_streaming(request: StartStreamingRequest):
    global transcription_thread, detection_thread, observer, streaming_client, streaming_manager, device_client, device, project_aria_handler

    args = argparse.Namespace(
        streaming_interface=request.interface,
        update_iptables=False,
        profile_name=request.profile_name,
    )
    project_aria_handler = ProjectARIAHandler(args)
    [streaming_client, streaming_manager, device_client, device] = (
        project_aria_handler.setup_project_aria()
    )

    try:
        streaming_manager.start_streaming()
        print(f"Streaming state: {streaming_manager.streaming_state}")

        return {"status": "Streaming started"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/stop_streaming")
async def stop_streaming():
    global transcription_thread, detection_thread, streaming_client, streaming_manager, device_client, device

    try:
        print("Stop listening to audio and image data")
        streaming_client.unsubscribe()
        print("Streaming unsubscribed")
        streaming_manager.stop_streaming()
        print("Streaming stopped")
        device_client.disconnect(device)
        print("Device disconnected")

        return {
            "status": "Streaming stopped",
            "transcriptions": transcriptions,
            "detected_mood": detected_mood,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/connect")
async def connect():
    database = DatabaseConnection(uri)
    try:
        connection = database.connect()
        print(connection)
        if connection:
            return {"status": "success", "message": "Connected to database"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Function to safely send text over websocket
async def send_text_safe(websocket: WebSocket, data: dict):
    try:
        await websocket.send_text(json.dumps(data))
    except Exception as e:
        print(f"Error sending WebSocket message: {e}")


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    global transcription_thread, detection_thread, observer, streaming_client, streaming_manager, device_client, device, project_aria_handler

    await websocket.accept()

    try:
        while True:
            observer = StreamingClientObserver(audio_queue, audio_processor, websocket)
            streaming_client.set_streaming_client_observer(observer)

            print("Start listening to audio and image data")
            streaming_client.subscribe()

            model = WhisperModel("tiny.en", device="cpu", compute_type="float32")
            transcription_thread = threading.Thread(
                target=audio_processor.transcribe_and_diarize_audio,
                args=(model, audio_queue, websocket),
                daemon=True,
            )
            transcription_thread.start()
            print("Transcription thread started")

            detection_thread = threading.Thread(
                target=observer.detect_mood,
                args=(detected_mood, websocket),
                daemon=True,
            )
            detection_thread.start()
            print("Mood detection thread started")

            await asyncio.Future()  # Run forever
    except WebSocketDisconnect:
        print("Client disconnected")
    finally:
        await websocket.close()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
