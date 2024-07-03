from pyannote.audio import Pipeline
from dotenv import load_dotenv
import os

load_dotenv()

AUTH_TOKEN = os.getenv("AUTH_TOKEN")

pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization@2.1", use_auth_token=AUTH_TOKEN
)

# apply the pipeline to an audio file
diarization = pipeline("./images/Sample.wav")
print("Diarization is completed.")
print(diarization)
