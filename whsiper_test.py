import numpy as np
import wave
from scipy.signal import resample
from faster_whisper import WhisperModel


# Function to read audio values from a text file
def read_audio_values_from_file(file_path):
    with open(file_path, "r") as f:
        data = f.read()
    audio_values = list(map(int, data.split(",")))

    return audio_values


# Function to convert audio values to a WAV file
def convert_to_wav(data, output_filename="output_32bit.wav"):
    audio_data = np.array(data, dtype=np.int32)
    # Find the maximum absolute value for normalization
    max_val = np.max(np.abs(audio_data))

    # Normalize data to fit within the 32-bit signed integer range
    normalized_data = (audio_data / max_val * (2**31 - 1)).astype(np.int32)

    # Set parameters for the WAV file
    sample_rate = 48000  # Typical sample rate for speech recognition
    n_channels = 7  # Mono audio
    sampwidth = 4  # 4 bytes per sample for 32-bit audio

    # Create the WAV file
    with wave.open(output_filename, "w") as wf:
        wf.setnchannels(n_channels)
        wf.setsampwidth(sampwidth)
        wf.setframerate(sample_rate)
        wf.writeframes(audio_data.tobytes())

    print(f"WAV file created: {output_filename}")


# Function to load a WAV file as a NumPy array
def load_wav_as_numpy(file_path):
    with wave.open(file_path, "rb") as wf:
        n_channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        sample_rate = wf.getframerate()
        n_frames = wf.getnframes()

        audio_data = wf.readframes(n_frames)
        audio_data = np.frombuffer(audio_data, dtype=np.int32)

        if n_channels > 1:
            audio_data = audio_data.reshape((-1, n_channels))
            audio_data = audio_data.mean(axis=1)

        return audio_data, sample_rate


# Function to transcribe audio using Whisper
def transcribe_audio(model, audio_data, sample_rate=50000):
    # Resample if needed
    if sample_rate != 16000:
        num_samples = int(len(audio_data) * 16000 / sample_rate)
        audio_data = resample(audio_data, num_samples)

    # Normalize to range -1 to 1
    audio_data = audio_data / np.max(np.abs(audio_data))

    # Convert to float32 for Whisper model
    audio_data = audio_data.astype(np.float32)

    # Transcribe using Whisper
    segments, info = model.transcribe(audio_data, beam_size=5, language="en")
    return segments, info


# Function to print transcription output
def print_transcription_output(segments):
    for segment in segments:
        print(f"Start: {segment.start:.2f} seconds")
        print(f"End: {segment.end:.2f} seconds")
        print(f"Text: {segment.text}")
        print(f"Tokens: {segment.tokens}")


# Main function
def main():
    # Path to the text file containing audio values
    text_file_path = "audio_values.txt"

    # Read audio values from the file
    audio_values = read_audio_values_from_file(text_file_path)

    # Convert audio values to a WAV file
    wav_file_path = "output_32bit.wav"
    convert_to_wav(audio_values, output_filename=wav_file_path)

    # Load the WAV file as a NumPy array
    audio_data, sample_rate = load_wav_as_numpy(wav_file_path)

    # Initialize the Whisper model
    model = WhisperModel("tiny.en", device="cpu", compute_type="int8")

    # Perform transcription
    segments, info = transcribe_audio(model, audio_values)

    # Print the transcription output
    print_transcription_output(segments)


if __name__ == "__main__":
    main()


# import numpy as np
# from scipy.signal import resample
# from faster_whisper import WhisperModel


# # Function to read audio values from a text file
# def read_audio_values_from_file(file_path):
#     with open(file_path, "r") as f:
#         data = f.read()
#     audio_values = list(map(int, data.split(",")))

#     return np.array(audio_values)


# # Function to process multi-channel audio data
# def process_multi_channel_audio(audio_data, num_channels):
#     total_samples = len(audio_data)

#     # Ensure the data length is a multiple of num_channels
#     if total_samples % num_channels != 0:
#         print(
#             f"Warning: Number of audio samples ({total_samples}) is not a multiple of number of channels ({num_channels}). Truncating excess samples."
#         )
#         truncated_samples = total_samples - (total_samples % num_channels)
#         audio_data = audio_data[:truncated_samples]

#     # Reshape the data into the given number of channels
#     try:
#         audio_data = audio_data.reshape(-1, num_channels)
#     except ValueError as e:
#         raise ValueError(f"Error reshaping audio data: {e}")

#     # Average the channels to convert to mono
#     mono_audio = np.mean(audio_data, axis=1)

#     return mono_audio


# # Function to transcribe audio using Whisper
# def transcribe_audio(
#     model, audio_data, original_sample_rate=50000, target_sample_rate=16000
# ):
#     # Resample to target sample rate
#     num_samples = int(len(audio_data) * target_sample_rate / original_sample_rate)
#     audio_data_resampled = resample(audio_data, num_samples)

#     # Normalize to range -1 to 1
#     audio_data_resampled = audio_data_resampled / np.max(np.abs(audio_data_resampled))

#     # Convert to float32 for Whisper model
#     audio_data_resampled = audio_data_resampled.astype(np.float32)

#     # Transcribe using Whisper
#     segments, info = model.transcribe(audio_data_resampled, beam_size=5, language="en")
#     return segments, info


# # Function to print transcription output
# def print_transcription_output(segments):

#     for segment in segments:
#         print(f"Start: {segment.start:.2f} seconds")
#         print(f"End: {segment.end:.2f} seconds")
#         print(f"Text: {segment.text}")
#         print(f"Tokens: {segment.tokens}")


# # Main function
# def main():
#     # Path to the text file containing audio values
#     text_file_path = "audio_values.txt"

#     # Parameters
#     num_of_channels = 7
#     sample_rate = 50000
#     sample_width = 4  # Assuming this is in bytes

#     # Read audio values from the file
#     audio_values = read_audio_values_from_file(text_file_path)

#     # Process multi-channel audio
#     mono_audio_data = process_multi_channel_audio(audio_values, num_of_channels)

#     # Initialize the Whisper model
#     model = WhisperModel("tiny.en", device="cpu", compute_type="int8")

#     # Perform transcription
#     segments, info = transcribe_audio(
#         model, mono_audio_data, original_sample_rate=sample_rate
#     )

#     # Print the transcription output
#     print_transcription_output(segments)


# if __name__ == "__main__":
#     main()
