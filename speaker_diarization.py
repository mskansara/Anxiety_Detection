from pyannote.audio import Pipeline

AUTH_TOKEN = "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"
pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization@2.1", use_auth_token=AUTH_TOKEN
)

# apply the pipeline to an audio file
diarization = pipeline("Sample.wav")

print(diarization)
