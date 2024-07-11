from faster_whisper import WhisperModel

model_size = "tiny.en"

model = WhisperModel(model_size, device="cpu", compute_type="float32")

segments, info = model.transcribe(
    "../clips/Testing_Clips/accent3.mp3", beam_size=5, language="en"
)

# Collect the transcription text from each segment
transcription = "\n".join(segment.text for segment in segments)
print(transcription)

# Save the transcription to a text file
output_file = "./Transcripts/accent3.txt"
with open(output_file, "w") as f:
    f.write(transcription)
