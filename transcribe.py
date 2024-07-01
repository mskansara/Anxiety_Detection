# audio_processing/transcriber.py
import os
import numpy as np
from typing import List
from faster_whisper import WhisperModel
from pydub import AudioSegment
from scipy.signal import resample
import wave


class AudioTranscriber:
    def __init__(
        self, model_path: str, device: str = "cpu", compute_type: str = "float32"
    ):
        self.model = WhisperModel(model_path, device=device, compute_type=compute_type)

    def load_audio_files(self, folder_path: str) -> List[str]:
        """
        Load all MP3 files from the specified folder.

        Args:
            folder_path (str): Path to the folder containing MP3 files.

        Returns:
            List[str]: List of paths to the MP3 files.
        """
        audio_files = [
            os.path.join(folder_path, f)
            for f in os.listdir(folder_path)
            if f.endswith(".wav")
        ]
        return audio_files

    def transcribe_audio(self, audio_file: str) -> str:
        """
        Transcribe the provided audio file using WhisperModel.

        Args:
            audio_file (str): Path to the MP3 file.

        Returns:
            str: Transcription text.
        """
        with wave.open(audio_file, "rb") as wf:
            n_channels = wf.getnchannels()
            sample_width = wf.getsampwidth()
            sample_rate = wf.getframerate()
            n_frames = wf.getnframes()
            audio_data = wf.readframes(n_frames)
            audio_data = np.frombuffer(audio_data, dtype=np.int32)

            # Resample if needed
        if sample_rate != 16000:
            num_samples = int(len(audio_data) * 16000 / sample_rate)
            audio_data = resample(audio_data, num_samples)

        # Normalize to range -1 to 1
        audio_data = audio_data / np.max(np.abs(audio_data))

        # Convert to float32 for Whisper model
        audio_data = audio_data.astype(np.float32)

        segments, _ = self.model.transcribe(audio_data, beam_size=5, language="en")
        transcription = "\n".join(segment.text for segment in segments)
        return transcription

    def save_transcription(self, transcription: str, output_path: str):
        """
        Save the transcription to a text file.

        Args:
            transcription (str): The transcription text.
            output_path (str): Path to the output text file.
        """
        with open(output_path, "w") as f:
            f.write(transcription)

    def process_folder(self, folder_path: str, output_folder: str):
        """
        Process all MP3 files in the specified folder, transcribe them, and save the transcriptions.

        Args:
            folder_path (str): Path to the folder containing MP3 files.
            output_folder (str): Path to the folder where transcription text files will be saved.
        """
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        audio_files = self.load_audio_files(folder_path)
        for audio_file in audio_files:
            file_name = os.path.basename(audio_file).rsplit(".", 1)[0]
            output_path = os.path.join(output_folder, f"{file_name}.txt")
            print(f"Processing: {audio_file}")
            transcription = self.transcribe_audio(audio_file)
            self.save_transcription(transcription, output_path)
            print(f"Transcription saved: {output_path}")


# main.py


def main():
    folder_path = "./train"
    output_folder = "./transcriptions"
    model_path = "tiny.en"
    device = "cpu"
    transcriber = AudioTranscriber(model_path=model_path, device=device)
    transcriber.process_folder(folder_path, output_folder)


if __name__ == "__main__":
    main()
