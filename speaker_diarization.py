from pyannote.audio import Pipeline
from dotenv import load_dotenv
import os
from faster_whisper import WhisperModel

load_dotenv()

AUTH_TOKEN = os.getenv("AUTH_TOKEN")


def label_transcription(self, diarization, transcription):
    labeled_segments = []
    for segment, _, speaker in diarization.itertracks(yield_label=True):
        start_time = segment.start
        end_time = segment.end
        text_segment = self.get_text_segment(transcription, start_time, end_time)
        labeled_segments.append(f"[{speaker}] {text_segment}")
    return "\n".join(labeled_segments)


pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization@2.1", use_auth_token=AUTH_TOKEN
)
model = WhisperModel("tiny.en", device="cpu", compute_type="float32")
segments, _ = model.transcribe("./images/Sample.wav", beam_size=5, language="en")
transcription = "\n".join(segment.text for segment in segments)

# apply the pipeline to an audio file
diarization = pipeline("./images/Sample.wav")
print("Diarization is completed.")
print(diarization)

labeled_transcription = label_transcription(diarization, transcription)
print(labeled_transcription)
