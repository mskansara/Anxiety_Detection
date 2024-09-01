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
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Depends
import asyncio
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import json
from vosk import Model, KaldiRecognizer
import pexpect
import re

from database_connection import DatabaseConnection

from auth import router as auth_router
from auth import get_current_doctor, TokenData
import shlex
from motor.motor_asyncio import AsyncIOMotorClient
from bson import ObjectId
from model import Doctor, Patient, Session, PyObjectId
from typing import List, Dict, Any
from datetime import datetime


# API Endpoints
# Load environment variables
load_dotenv()
AUTH_TOKEN = os.getenv("AUTH_TOKEN")
MONGODB_AUTH = os.getenv("MONGODB_AUTH")
# uri = f"mongodb+srv://manthankansara7:{MONGODB_AUTH}@cluster0.4ubii7g.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
SSH_PASSPHRASE = os.getenv("SSH_PASSPHRASE")
uri = os.getenv("DB_URL")
# MongoDB connection
client = AsyncIOMotorClient(uri)
db = client[os.getenv("DB_NAME")]
print(client)

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
# Include the auth router so the /token endpoint is available
app.include_router(auth_router)


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


class ImageProcessor:
    def __init__(
        self,
    ) -> None:
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        self.image_counter = 0

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


class AudioProcessor:
    def __init__(self):
        self.lock = threading.Lock()

    def process_multi_channel_audio(self, audio_data, num_channels):
        # print("Processing multi-channel audio")
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

    def transcribe_audio(self, model, audio_queue, websocket):
        # print("Transcribing audio")
        target_chunk_size = int(TARGET_SAMPLE_RATE * BUFFER_DURATION)
        recognizer = KaldiRecognizer(model, TARGET_SAMPLE_RATE)

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
                # Convert the float32 audio data to int16 PCM data
                # audio_data = np.array(audio_data, dtype=np.float32)
                # audio_data = (audio_data * 32767).astype(np.int16)
                # byte_data = audio_data.tobytes()
                # # Only send the final transcription result
                # if recognizer.AcceptWaveform(byte_data):
                #     result = recognizer.Result()
                #     transcription = json.loads(result)["text"]

                #     data = {
                #         "transcription": transcription,
                #         "start_timestamp": start_audio_timestamp,
                #         "end_timestamp": end_audio_timestamp,
                #         "mood": None,
                #     }

                #     # Send the final transcription via WebSocket
                #     print(data)
                #     asyncio.run(send_text_safe(websocket, data))
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

            except Exception as e:
                print(f"Error during transcription: {e}")
            finally:
                time.sleep(0.1)


class StreamingClientObserver:
    def __init__(self, audio_queue, audio_processor):
        self.start_audio_timestamp = None
        self.end_audio_timestamp = None
        self.audio_queue = audio_queue
        self.audio_buffer = np.array([], dtype=np.float32)
        self.sample_rate = SAMPLE_RATE
        self.lock = threading.Lock()
        self.audio_processor = audio_processor
        self.rgb_image = {"data": None, "timestamp": None}

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


# Initialize global variables for threading and streaming
audio_queue = Queue()
audio_processor = AudioProcessor()
image_processor = ImageProcessor()
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


# start_streaming and stop_streaming should only be accessible by doctors
@app.post("/start_streaming")
async def start_streaming(
    request: StartStreamingRequest,
    current_user: TokenData = Depends(get_current_doctor),
):
    global transcription_thread, detection_thread, observer, streaming_client, streaming_manager, device_client, device, project_aria_handler

    args = parse_args()
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
async def stop_streaming(current_user: TokenData = Depends(get_current_doctor)):
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
            observer = StreamingClientObserver(audio_queue, audio_processor)
            streaming_client.set_streaming_client_observer(observer)

            print("Start listening to audio and image data")
            streaming_client.subscribe()

            model = WhisperModel("tiny.en", device="cpu", compute_type="float32")
            transcription_thread = threading.Thread(
                target=audio_processor.transcribe_audio,
                args=(model, audio_queue, websocket),
                daemon=True,
            )
            transcription_thread.start()
            print("Transcription thread started")

            detection_thread = threading.Thread(
                target=image_processor.detect_mood,
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


TIMEOUT = 60000  # Timeout in seconds


def establish_ssh_tunnel():
    try:
        # Step 1: Establish the SSH tunnel
        print("Establishing SSH tunnel...")
        ssh_command = "ssh -t msk2@csgate.ucc.ie ssh msk2@csg25-04.ucc.ie"

        child = pexpect.spawn(ssh_command, timeout=TIMEOUT, encoding="utf-8")
        child.expect("Enter passphrase for key '/Users/manthankansara/.ssh/id_rsa':")
        child.sendline(SSH_PASSPHRASE)

        child.expect("Enter passphrase for key '/users/msccs2024/msk2/.ssh/id_rsa':")
        child.sendline(SSH_PASSPHRASE)

        # After successful authentication, you should get a shell prompt
        child.expect(r"\$")

        print("SSH tunnel established successfully.")
        return child
    except pexpect.exceptions.EOF as e:
        print(f"Unexpected EOF during SSH tunnel establishment: {str(e)}")
        child.close()
        raise HTTPException(status_code=500, detail="SSH connection failed (EOF)")
    except pexpect.exceptions.TIMEOUT as e:
        print(f"Timeout during SSH tunnel establishment: {str(e)}")
        child.close()
        raise HTTPException(status_code=500, detail="SSH connection failed (timeout)")


def prettify_output(raw_output):
    # Replace escape sequences for newlines with actual newlines
    formatted_output = raw_output.replace("\\n", "\n")

    # Replace double backslashes with single backslashes
    formatted_output = formatted_output.replace("\\\\", "\\")

    # Remove unnecessary parts (like the extra escape sequences)
    formatted_output = formatted_output.replace("```python", "").replace("```", "")

    # Optional: Add additional formatting if needed (e.g., indentation)
    formatted_output = formatted_output.replace(
        "\\t", "    "
    )  # Replace tabs with spaces

    # Return the formatted output
    return formatted_output.strip()


def run_command(child, command):
    try:
        print(f"Running command: {command}")
        child.sendline(command)

        child.expect(r"\$", timeout=TIMEOUT)
        output = child.before

        # Print the extracted key points
        # print(final_output)
        return output
    except pexpect.exceptions.TIMEOUT as e:
        print(f"Timeout during command execution: {str(e)}")
        child.close()
        raise HTTPException(
            status_code=500, detail="Command execution failed (timeout)"
        )
    except pexpect.exceptions.EOF as e:
        print(f"Unexpected EOF during command execution: {str(e)}")
        child.close()
        raise HTTPException(status_code=500, detail="Command execution failed (EOF)")


@app.get("/generateSummary")
async def generateSummary():
    try:
        session_id = "66d0602937f3f9caafb68345"
        curl_command = f"""curl -X POST http://localhost:8000/generate -H "Content-Type: application/json" -d '{{\"session_id\": \"{session_id}\"}}'"""

        # Establish the SSH tunnel (this is where ssh_tunneling is happening)
        child = establish_ssh_tunnel()

        # Run the curl command on the remote server
        output = run_command(child, curl_command)
        print(output)
        # Close the SSH session
        child.sendline("exit")
        child.close()
        # print(f"Command output: {output}")
        start = output.find("summary:")
        end = output.find("```python")
        # Extract the "Key points" section

        key_points_section = output[start:end].strip()
        final_output = prettify_output(key_points_section)
        print(final_output)
        try:
            summary_response = json.loads(final_output)
        except json.JSONDecodeError:
            raise HTTPException(
                status_code=500, detail="Failed to decode JSON response"
            )

        # Update the session document with the generated summary
        result = await db.sessions.update_one(
            {"_id": ObjectId(session_id)},
            {"$set": {"summary": final_output}},
        )

        if result.modified_count == 0:
            raise HTTPException(
                status_code=404, detail="Session not found or not updated"
            )

        # Fetch the updated session document to return to the frontend
        updated_session = await db.sessions.find_one({"_id": ObjectId(session_id)})
        if updated_session is None:
            raise HTTPException(
                status_code=404, detail="Session not found after update"
            )

        return {"output": final_output}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Command failed: {str(e)}")


@app.post("/doctors/", response_model=Doctor)
async def create_doctor(doctor: Doctor):
    doctor = doctor.dict(by_alias=True)
    result = await db.doctors.insert_one(doctor)
    doctor["_id"] = result.inserted_id
    return doctor


@app.post("/patients/", response_model=Patient)
async def create_patient(patient: Patient):
    patient = patient.dict(by_alias=True)
    result = await db.patients.insert_one(patient)
    patient["_id"] = result.inserted_id
    return patient


class CreateSessionRequest(BaseModel):
    doctor_id: str
    patient_id: str


@app.post("/sessions")
async def create_session(session_data: CreateSessionRequest):
    try:
        session = {
            "doctor_id": ObjectId(session_data.doctor_id),
            "patient_id": ObjectId(session_data.patient_id),
            "session_date": datetime.now(),
            "session_data": {"transcriptions": [], "detected_mood": []},
            "summary": "",
        }
        result = await db.sessions.insert_one(session)

        if result.inserted_id:
            return {"session_id": str(result.inserted_id)}
        else:
            raise HTTPException(status_code=500, detail="Failed to create session")
    except Exception as e:
        print(f"Error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal Server Error")


@app.get("/doctors/{doctor_id}", response_model=Doctor)
async def get_doctor(doctor_id: str):
    doctor = await db.doctors.find_one({"_id": ObjectId(doctor_id)})
    if doctor is None:
        raise HTTPException(status_code=404, detail="Doctor not found")
    return doctor


@app.get("/patients/{patient_id}", response_model=Patient)
async def get_patient(patient_id: str):
    patient = await db.patients.find_one({"_id": ObjectId(patient_id)})
    if patient is None:
        raise HTTPException(status_code=404, detail="Patient not found")
    return patient


@app.get("/sessions/{session_id}", response_model=Session)
async def get_session(session_id: str):
    session = await db.sessions.find_one({"_id": ObjectId(session_id)})
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return session


class UpdateSessionRequest(BaseModel):
    session_id: str
    transcriptions: List[Dict[str, Any]]
    detected_mood: List[Dict[str, Any]]


@app.post("/sessions/update")
async def update_session(session_data: UpdateSessionRequest):
    try:
        # Convert session_id to ObjectId
        print(session_data)
        session_id = ObjectId(session_data.session_id)

        # Update the session document in the database
        update_data = {
            "$set": {
                "session_data.transcriptions": session_data.transcriptions,
                "session_data.detected_mood": session_data.detected_mood,
            }
        }

        result = await db.sessions.update_one({"_id": session_id}, update_data)

        if result.modified_count == 1:
            return {"status": "success", "message": "Session updated successfully"}
        else:
            raise HTTPException(
                status_code=404, detail="Session not found or data not modified"
            )
    except Exception as e:
        print(f"Error updating session: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal Server Error")


class DoctorIDRequest(BaseModel):
    doctor_id: str


@app.post("/doctors/patients", response_model=List[Patient])
async def get_patients_by_doctor(request: DoctorIDRequest):
    doctor = await db.doctors.find_one({"_id": ObjectId(request.doctor_id)})
    if doctor is None:
        raise HTTPException(status_code=404, detail="Doctor not found")

    # Fetch patients using the list of patient ObjectIds from the doctor's document
    patient_ids = doctor.get("patients", [])
    patients = await db.patients.find({"_id": {"$in": patient_ids}}).to_list(
        length=None
    )

    return [Patient(**patient) for patient in patients]


@app.get("/doctors", response_model=List[Doctor])
async def get_all_doctors():
    doctors_cursor = db.doctors.find()
    doctors = await doctors_cursor.to_list(length=None)
    # print(doctors)
    if not doctors:
        print("No doctors found")  # Debug output
    else:
        print(f"Found {len(doctors)} doctors")  # Debug output
    return doctors


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
