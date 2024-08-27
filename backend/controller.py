# controller.py

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
from bson import ObjectId
from typing import List
from models import Doctor, Patient, Session, PyObjectId
from project_aria_handler import (
    ProjectARIAHandler,
    AudioProcessor,
    StreamingClientObserver,
    StartStreamingRequest,
    send_text_safe,
)
import threading
import asyncio
from queue import Queue
import json
from vosk import Model
import pexpect
import os

# Global variables
audio_queue = Queue()
audio_processor = AudioProcessor()
observer = None
transcription_thread = None
detection_thread = None
streaming_client = None
streaming_manager = None
device_client = None
device = None

# Load environment variables
MONGODB_AUTH = os.getenv("MONGODB_AUTH")
uri = f"mongodb+srv://manthankansara7:{MONGODB_AUTH}@cluster0.4ubii7g.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"

# Initialize FastAPI app
app = FastAPI()

# CORS setup for ReactJS frontend
origins = ["http://localhost:3000", "http://localhost"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# MongoDB connection
client = AsyncIOMotorClient(uri)
db = client.mydatabase

# API Endpoints


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


@app.post("/sessions/", response_model=Session)
async def create_session(session: Session):
    session = session.dict(by_alias=True)
    result = await db.sessions.insert_one(session)
    session["_id"] = result.inserted_id
    return session


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


@app.post("/start_streaming")
async def start_streaming(request: StartStreamingRequest):
    global transcription_thread, detection_thread, observer, streaming_client, streaming_manager, device_client, device, project_aria_handler

    project_aria_handler = ProjectARIAHandler(request)
    [streaming_client, streaming_manager, device_client, device] = (
        project_aria_handler.setup_project_aria()
    )

    try:
        streaming_manager.start_streaming()
        return {"status": "Streaming started"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/stop_streaming")
async def stop_streaming():
    global transcription_thread, detection_thread, streaming_client, streaming_manager, device_client, device

    try:
        streaming_client.unsubscribe()
        streaming_manager.stop_streaming()
        device_client.disconnect(device)
        return {"status": "Streaming stopped"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/connect")
async def connect():
    try:
        connection = await client.server_info()  # Test connection
        if connection:
            return {"status": "success", "message": "Connected to database"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    global transcription_thread, detection_thread, observer, streaming_client, streaming_manager, device_client, device, project_aria_handler

    await websocket.accept()

    try:
        observer = StreamingClientObserver(audio_queue, audio_processor, websocket)
        streaming_client.set_streaming_client_observer(observer)

        streaming_client.subscribe()

        # Load Vosk model
        model = Model("./models/vosk-model-en-us-0.42-gigaspeech")
        transcription_thread = threading.Thread(
            target=audio_processor.transcribe_and_diarize_audio,
            args=(model, audio_queue, websocket),
            daemon=True,
        )
        transcription_thread.start()

        detection_thread = threading.Thread(
            target=observer.detect_mood,
            args=(detected_mood, websocket),
            daemon=True,
        )
        detection_thread.start()

        await asyncio.Future()  # Run forever
    except WebSocketDisconnect:
        print("Client disconnected")
    finally:
        await websocket.close()


@app.get("/execute_ssh_command")
async def execute_ssh_command():
    try:
        command = "ssh -t msk2@csgate.ucc.ie ssh msk2@csg25-04.ucc.ie curl http://localhost:8000/"
        password = "CISORIg5"

        child = pexpect.spawn(command)
        child.expect("password:")
        child.sendline(password)
        child.expect(pexpect.EOF)
        output = child.before.decode("utf-8")

        return {"output": output}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Command failed: {str(e)}")
