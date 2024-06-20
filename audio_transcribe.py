import numpy as np
import argparse
import sys
import aria.sdk as aria
import time
from queue import Queue, Empty
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
SAMPLE_RATE = 48000  # Original sample rate from live streaming
TARGET_SAMPLE_RATE = 16000  # Target sample rate for Whisper
NUM_CHANNELS = 7  # Number of audio channels
BUFFER_DURATION = 10  # Buffer duration for processing in seconds
CHUNK_DURATION = 10  # Chunk duration for transcription in seconds


# Function to handle real-time transcription
def transcribe_audio(model, audio_queue, lock):
    CHUNK_DURATION = (
        BUFFER_DURATION  # Define chunk duration same as buffer duration for clarity
    )
    target_chunk_size = int(TARGET_SAMPLE_RATE * CHUNK_DURATION)
    print(f"Target chunk size for processing: {target_chunk_size} samples")

    while True:
        try:
            audio_data = []

            while len(audio_data) < target_chunk_size:
                lock.acquire()
                try:
                    # Calculate how much more data we need to reach the target chunk size
                    needed_samples = target_chunk_size - len(audio_data)

                    if not audio_queue.empty():
                        # Get a chunk of audio data from the queue
                        queue_data = audio_queue.get(timeout=1)
                        # Check how much data we can take from this chunk
                        chunk_data = queue_data[:needed_samples]

                        # Add the chunk to our audio_data
                        audio_data.extend(chunk_data)
                        # print(f"Retrieved {len(chunk_data)} samples from queue.")
                        # print(f"Current audio_data length: {len(audio_data)} samples")

                        # If there are leftover samples, put them back into the queue for future processing
                        if len(queue_data) > needed_samples:
                            remaining_data = queue_data[needed_samples:]
                            audio_queue.put(remaining_data)
                            print(f"Leftover {len(remaining_data)} samples re-queued.")
                    # else:
                    #     print("")
                    #     # print("Queue is empty, waiting for data...")
                finally:
                    lock.release()

                # Sleep briefly to avoid busy waiting
                time.sleep(0.1)

            if audio_data:
                # print(f"Processing {len(audio_data)} samples for transcription.")
                audio_data = np.array(audio_data, dtype=np.float32)
                # print(audio_data)
                segments, _ = model.transcribe(audio_data, beam_size=5, language="en")
                for segment in segments:
                    print(f"{segment.text}")
                # with open("transcriptions.txt", "a") as f:
                #     for segment in segments:
                #         f.write(f"{segment.text}\n")
                #         print(
                #             f"[{segment.start:.2f} - {segment.end:.2f}]: {segment.text}"
                #         )

            else:
                print("No audio data to process.")

        except Exception as e:
            print(f"Error during transcription: {e}")
        finally:
            time.sleep(0.1)


# Function to convert 32-bit multi-channel audio data to mono 16-bit
def process_multi_channel_audio(audio_data, num_channels):
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
        self.audio_buffer = np.array([], dtype=np.float32)
        self.sample_rate = SAMPLE_RATE
        self.lock = threading.Lock()

    def on_audio_received(self, audio_data: AudioData, record: AudioDataRecord):
        audio_data_values = np.array(audio_data.data, dtype=np.float32)
        num_samples_per_channel = len(audio_data_values) // NUM_CHANNELS

        # Reshape and process multi-channel audio to mono
        audio_data_values = audio_data_values[: num_samples_per_channel * NUM_CHANNELS]
        mono_audio = process_multi_channel_audio(audio_data_values, NUM_CHANNELS)

        # Append new audio to buffer
        self.audio_buffer = np.concatenate((self.audio_buffer, mono_audio))

        # Process audio buffer if it has enough data for the buffer duration
        buffer_sample_count = int(SAMPLE_RATE * BUFFER_DURATION)
        if len(self.audio_buffer) >= buffer_sample_count:
            buffer_audio = self.audio_buffer[:buffer_sample_count]
            self.audio_buffer = self.audio_buffer[buffer_sample_count:]

            # Resample to target sample rate for Whisper
            resampled_audio = resample_audio(
                buffer_audio, SAMPLE_RATE, TARGET_SAMPLE_RATE
            )

            # Normalize audio to -1, 1 and convert to float32
            normalized_audio = normalize_audio(resampled_audio)

            # Add to queue
            self.lock.acquire()
            self.audio_queue.put(normalized_audio.tolist())
            self.lock.release()


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

        model = WhisperModel("tiny.en", device="cpu", compute_type="float32")
        transcription_thread = threading.Thread(
            target=transcribe_audio,
            args=(model, audio_queue, observer.lock),
            daemon=True,
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
