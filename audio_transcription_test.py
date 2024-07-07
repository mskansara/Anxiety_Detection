import numpy as np
import argparse
import sys
import aria.sdk as aria
import time
import threading
from faster_whisper import WhisperModel
from projectaria_tools.core.sensor_data import AudioData, AudioDataRecord

# from pyannote.audio import Pipeline
import os
from dotenv import load_dotenv
import torch


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


# Constants
SAMPLE_RATE = 48000  # Original sample rate from live streaming
TARGET_SAMPLE_RATE = 16000  # Target sample rate for Whisper
NUM_CHANNELS = 7  # Number of audio channels
BUFFER_DURATION = 10  # Buffer duration for processing in seconds

load_dotenv()

AUTH_TOKEN = os.getenv("AUTH_TOKEN")


class AudioProcessor:
    # def __init__(self):
    # self.pipeline = Pipeline.from_pretrained(
    #     "pyannote/speaker-diarization@2.1",
    #     use_auth_token=AUTH_TOKEN,
    # )

    # Function to convert 32-bit multi-channel audio data to mono 16-bit
    def process_multi_channel_audio(self, audio_data, num_channels):
        total_samples = len(audio_data)

        # Ensure the data length is a multiple of num_channels
        if total_samples % num_channels != 0:
            truncated_samples = total_samples - (total_samples % num_channels)
            audio_data = audio_data[:truncated_samples]

        # Reshape the data into the given number of channels
        audio_data = audio_data.reshape(-1, num_channels)

        # Average the channels to convert to mono
        mono_audio = np.mean(audio_data, axis=1)

        return mono_audio

    # Function to resample audio to the target sample rate
    def resample_audio(self, audio_data, original_sample_rate, target_sample_rate):
        num_samples = int(len(audio_data) * target_sample_rate / original_sample_rate)
        resampled_data = np.interp(
            np.linspace(0, len(audio_data), num_samples),
            np.arange(len(audio_data)),
            audio_data,
        )
        return resampled_data.astype(np.float32)

    # Function to normalize audio data to -1, 1 range
    def normalize_audio(self, audio_data):
        max_val = np.max(np.abs(audio_data))
        if max_val > 0:
            normalized_data = audio_data / max_val
        else:
            normalized_data = audio_data
        return normalized_data

    # Function to handle real-time transcription and diarization
    def transcribe_and_diarize_audio(self, model, audio_data, output_folder):
        try:
            # Convert the audio data to a numpy array
            audio_data = np.array(audio_data, dtype=np.float32)

            # Transcribe the audio data
            segments, _ = model.transcribe(audio_data, beam_size=5, language="en")
            transcription = "\n".join(segment.text for segment in segments)
            if transcription:
                print(transcription)
            # Prepare audio for diarization
            # waveform = torch.tensor([audio_data], dtype=torch.float32)
            # print(waveform)

            # # Diarize the audio
            # diarization = self.pipeline(
            #     {"waveform": waveform, "sample_rate": TARGET_SAMPLE_RATE}
            # )
            # print(diarization)
            # # Match transcription with diarization
            # labeled_transcription = self.label_transcription(diarization, transcription)
            # print(labeled_transcription)
            # self.save_transcription(labeled_transcription, output_folder)
        except Exception as e:
            print(f"Error during transcription and diarization: {e}")

    def label_transcription(self, diarization, transcription):
        labeled_segments = []
        for segment, _, speaker in diarization.itertracks(yield_label=True):
            start_time = segment.start
            end_time = segment.end
            text_segment = self.get_text_segment(transcription, start_time, end_time)
            labeled_segments.append(f"[{speaker}] {text_segment}")
        return "\n".join(labeled_segments)

    def get_text_segment(self, transcription, start_time, end_time):
        # Placeholder to match the text segment with the time range.
        text_segment = transcription[int(start_time) : int(end_time)]
        return text_segment

    def save_transcription(self, transcription, output_folder):
        output_path = os.path.join(output_folder, "transcription.txt")
        with open(output_path, "w") as f:
            f.write(transcription)


class StreamingClientObserver:
    def __init__(self, audio_processor, model, output_folder):
        self.audio_processor = audio_processor
        self.model = model
        self.output_folder = output_folder

    def on_audio_received(self, audio_data: AudioData, record: AudioDataRecord):
        try:
            # Process the incoming audio data
            audio_data_values = np.array(audio_data.data, dtype=np.float32)
            print(audio_data_values.shape)
            num_samples_per_channel = len(audio_data_values) // NUM_CHANNELS

            # Reshape and process multi-channel audio to mono
            audio_data_values = audio_data_values[
                : num_samples_per_channel * NUM_CHANNELS
            ]
            mono_audio = self.audio_processor.process_multi_channel_audio(
                audio_data_values, NUM_CHANNELS
            )
            print(mono_audio.shape)

            # Resample and normalize audio
            resampled_audio = self.audio_processor.resample_audio(
                mono_audio, SAMPLE_RATE, TARGET_SAMPLE_RATE
            )
            print(resampled_audio.shape)
            normalized_audio = self.audio_processor.normalize_audio(resampled_audio)

            # Perform transcription and diarization
            self.audio_processor.transcribe_and_diarize_audio(
                self.model, resampled_audio, self.output_folder
            )
        except Exception as e:
            print(f"Error processing audio: {e}")


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
        if self.args.device_ip:
            client_config.ip_v4_address = self.args.device_ip
        self.device_client.set_client_config(client_config)
        device = self.device_client.connect()

        self.streaming_manager = device.streaming_manager
        self.streaming_client = self.streaming_manager.streaming_client

        streaming_config = aria.StreamingConfig()
        streaming_config.profile_name = self.args.profile_name
        if self.args.streaming_interface == "usb":
            streaming_config.streaming_interface = aria.StreamingInterface.Usb
        streaming_config.security_options.use_ephemeral_certs = True
        self.streaming_manager.streaming_config = streaming_config
        config = self.streaming_client.subscription_config
        config.subscriber_data_type = aria.StreamingDataType.Audio
        config.message_queue_size[aria.StreamingDataType.Audio] = 10000
        config.security_options.use_ephemeral_certs = True
        self.streaming_client.subscription_config = config

        return [
            self.streaming_client,
            self.streaming_manager,
            self.device_client,
            device,
        ]

    def update_iptables(self):
        # Implement iptables update logic here if needed for Linux
        pass


def main():
    output_folder = "./transcriptions"
    args = parse_args()
    project_aria_handler = ProjectARIAHandler(args)
    [streaming_client, streaming_manager, device_client, device] = (
        project_aria_handler.setup_project_aria()
    )
    try:
        streaming_manager.start_streaming()
        print(f"Streaming state: {streaming_manager.streaming_state}")

        # Initialize the audio processor and Whisper model
        audio_processor = AudioProcessor()
        model = WhisperModel("tiny.en", device="cpu", compute_type="float32")

        # Create the observer and start listening to audio data
        observer = StreamingClientObserver(audio_processor, model, output_folder)
        streaming_client.set_streaming_client_observer(observer)

        print("Start listening to audio data")
        streaming_client.subscribe()

        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Exiting...")
    finally:
        print("Stop listening to audio data")
        streaming_client.unsubscribe()
        streaming_manager.stop_streaming()
        device_client.disconnect(device)


if __name__ == "__main__":
    main()
