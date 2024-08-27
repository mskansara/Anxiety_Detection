from pymongo import MongoClient
from bson import ObjectId
from datetime import datetime
import random
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()
DB_URL = os.getenv("DB_URL")
DB_NAME = os.getenv("DB_NAME")

# MongoDB connection
client = MongoClient(DB_URL)
db = client[DB_NAME]
patients_collection = db["patients"]
doctors_collection = db["doctors"]
sessions_collection = db["sessions"]

# Sample data for patients and doctors
patient_names = [
    "John Doe",
    "Jane Smith",
    "Alice Johnson",
    "Bob Brown",
    "Charlie Davis",
]
doctor_names = [
    "Emily White",
    "Frank Green",
    "George Black",
    "Hannah Grey",
    "Irene Blue",
]
specializations = [
    "Psychiatrist",
    "Psychologist",
    "Therapist",
    "Counselor",
    "Neuroscientist",
]


def generate_unique_email(collection, base_email):
    """Generate a unique email address by appending a number if necessary."""
    email = base_email
    counter = 1
    while collection.find_one({"contact.email": email}):
        email = f"{base_email.split('@')[0]}{counter}@gmail.com"
        counter += 1
    return email


def generate_email(name):
    """Generate an email based on the name."""
    first_name, last_name = name.split()
    return f"{first_name.lower()}{last_name.lower()}@gmail.com"


# Insert dummy patients
patient_ids = []
for i in range(10):
    name = random.choice(patient_names)
    base_email = generate_email(name)
    email = generate_unique_email(patients_collection, base_email)

    patient_data = {
        "name": name,
        "dob": datetime(1990, 1, i + 1),
        "contact": {"phone": f"+1234567890{i}", "email": email},
        "doctors": [],  # Will be populated after doctor insertion
    }

    patient_id = patients_collection.insert_one(patient_data).inserted_id
    patient_ids.append(patient_id)

# Insert dummy doctors
doctor_ids = []
for i in range(10):
    name = random.choice(doctor_names)
    base_email = generate_email(name)
    email = generate_unique_email(doctors_collection, base_email)

    doctor_data = {
        "name": f"Dr. {name}",
        "specialization": random.choice(specializations),
        "contact": {"phone": f"+0987654321{i}", "email": email},
        "patients": patient_ids,  # Associate all patients with each doctor for simplicity
    }
    doctor_id = doctors_collection.insert_one(doctor_data).inserted_id
    doctor_ids.append(doctor_id)

# Sample transcription and detected mood data
sample_transcription = [
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
]

sample_mood = [
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
]

# Insert dummy sessions
for i in range(10):
    session_data = {
        "transcriptions": sample_transcription,
        "detected_mood": sample_mood,
    }
    session_data_to_insert = {
        "patient_id": random.choice(patient_ids),
        "doctor_id": random.choice(doctor_ids),
        "session_date": datetime.now(),
        "session_data": session_data,
        "summary": "This is a summary of the session detailing the patient's mood and discussion points.",
    }
    sessions_collection.insert_one(session_data_to_insert)

print("Dummy data inserted successfully.")
