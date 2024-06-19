import numpy as np
import argparse
import sys
import aria.sdk as aria
import wave
import io
import time
from queue import Queue
import threading
from faster_whisper import WhisperModel
from projectaria_tools.core.sensor_data import AudioData, AudioDataRecord


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
SAMPLE_RATE = 50000  # Original sample rate from live streaming
TARGET_SAMPLE_RATE = 16000  # Target sample rate for Whisper
NUM_CHANNELS = 7  # Number of audio channels
BUFFER_DURATION = 5  # Buffer duration for processing in seconds


# Function to handle real-time transcription
def transcribe_audio(model, audio_queue):
    while True:
        if not audio_queue.empty():
            audio_data = []
            while not audio_queue.empty():
                audio_data.extend(audio_queue.get())
                # print(audio_data)

            # Convert to numpy array
            audio_data = np.array(audio_data)

            # Process audio data into 5-second chunks
            total_samples = len(audio_data)
            chunk_samples = TARGET_SAMPLE_RATE * BUFFER_DURATION
            num_chunks = total_samples // chunk_samples

            for i in range(num_chunks):
                chunk_start = i * chunk_samples
                chunk_end = chunk_start + chunk_samples
                audio_chunk = audio_data[chunk_start:chunk_end]

                segments, _ = model.transcribe(audio_data, beam_size=5, language="en")
                for segment in segments:
                    print(f"[{segment.start:.2f} - {segment.end:.2f}]: {segment.text}")

        else:
            time.sleep(0.1)


# Function to convert 32-bit multi-channel audio data to mono 16-bit
def process_multi_channel_audio(audio_data, num_channels):
    total_samples = len(audio_data)

    # Ensure the data length is a multiple of num_channels
    if total_samples % num_channels != 0:
        print(
            f"Warning: Number of audio samples ({total_samples}) is not a multiple of number of channels ({num_channels}). Truncating excess samples."
        )
        truncated_samples = total_samples - (total_samples % num_channels)
        truncated_data = audio_data[truncated_samples:]  # Save the truncated part
        audio_data = audio_data[:truncated_samples]

        # Reshape the data into the given number of channels
        try:
            audio_data = audio_data.reshape(-1, num_channels)
        except ValueError as e:
            raise ValueError(f"Error reshaping audio data: {e}")

        # Average the channels to convert to mono
        mono_audio = np.mean(audio_data, axis=1)

        return mono_audio, truncated_data
    else:
        # Reshape the data into the given number of channels
        try:
            audio_data = audio_data.reshape(-1, num_channels)
        except ValueError as e:
            raise ValueError(f"Error reshaping audio data: {e}")

        # Average the channels to convert to mono
        mono_audio = np.mean(audio_data, axis=1)

        return mono_audio, np.array([])  # No truncated data


# Function to resample audio to the target sample rate
def resample_audio(audio_data, original_sample_rate, target_sample_rate):
    num_samples = int(len(audio_data) * target_sample_rate / original_sample_rate)
    resampled_data = np.interp(
        np.linspace(0, len(audio_data), num_samples),
        np.arange(len(audio_data)),
        audio_data,
    )
    return resampled_data.astype(np.float32)


# Function to normalize audio data to -1, 1 range
def normalize_audio(audio_data):
    max_val = np.max(np.abs(audio_data))
    if max_val > 0:
        normalized_data = audio_data / max_val
    else:
        normalized_data = audio_data
    return normalized_data


class StreamingClientObserver:
    def __init__(self, audio_queue):
        self.audio_queue = audio_queue
        self.audio_buffer = []
        self.truncated_data = []  # Buffer to keep truncated data
        self.sample_rate = 50000

    def on_audio_received(self, audio_data: AudioData, record: AudioDataRecord):
        audio_data_values = audio_data.data
        num_samples_per_channel = len(audio_data_values) // NUM_CHANNELS

        # Process each sample and channel in a nested loop
        for i in range(num_samples_per_channel):
            for c in range(NUM_CHANNELS):
                index = i * NUM_CHANNELS + c
                if index < len(audio_data_values):
                    self.audio_buffer.append(audio_data_values[index])

        # Combine previous truncated data with new incoming data
        self.audio_buffer.extend(self.truncated_data)

        # Calculate the buffer duration in seconds
        current_buffer_duration = len(self.audio_buffer) / (SAMPLE_RATE * NUM_CHANNELS)

        if current_buffer_duration >= BUFFER_DURATION:
            mono_audio, self.truncated_data = process_multi_channel_audio(
                np.array(self.audio_buffer), NUM_CHANNELS
            )
            self.audio_buffer = (
                self.truncated_data.tolist()
            )  # Keep truncated part for next round

            # Resample to target sample rate for Whisper
            resampled_audio = resample_audio(
                mono_audio, SAMPLE_RATE, TARGET_SAMPLE_RATE
            )

            # Normalize audio to -1, 1 and convert to float32
            normalized_audio = normalize_audio(resampled_audio)

            self.audio_queue.put(
                normalized_audio.tolist()
            )  # Convert to list before queuing


# Main function to set up and manage streaming
def main():
    args = parse_args()
    if args.update_iptables and sys.platform.startswith("linux"):
        update_iptables()

    aria.set_log_level(aria.Level.Info)
    device_client = aria.DeviceClient()
    client_config = aria.DeviceClientConfig()
    if args.device_ip:
        client_config.ip_v4_address = args.device_ip
    device_client.set_client_config(client_config)
    device = device_client.connect()

    streaming_manager = device.streaming_manager
    streaming_client = streaming_manager.streaming_client

    streaming_config = aria.StreamingConfig()
    streaming_config.profile_name = args.profile_name
    if args.streaming_interface == "usb":
        streaming_config.streaming_interface = aria.StreamingInterface.Usb
    streaming_config.security_options.use_ephemeral_certs = True
    streaming_manager.streaming_config = streaming_config

    try:
        streaming_manager.start_streaming()
        print(f"Streaming state: {streaming_manager.streaming_state}")

        config = streaming_client.subscription_config
        config.subscriber_data_type = aria.StreamingDataType.Audio
        config.message_queue_size[aria.StreamingDataType.Audio] = 100
        config.security_options.use_ephemeral_certs = True
        streaming_client.subscription_config = config

        audio_queue = Queue()
        observer = StreamingClientObserver(audio_queue)
        streaming_client.set_streaming_client_observer(observer)

        print("Start listening to audio data")
        streaming_client.subscribe()

        model = WhisperModel("tiny.en", device="cpu", compute_type="int8")
        # Try this method later.
        # while True:
        #     if not audio_queue.empty():
        #         audio_data = audio_queue.get()
        #         transcribe_audio(model, audio_data)
        #     else:
        #         time.sleep(0.1)  # Adjust sleep time as needed for your application

        transcription_thread = threading.Thread(
            target=transcribe_audio, args=(model, audio_queue), daemon=True
        )
        transcription_thread.start()
        print("Transcription thread started")

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
