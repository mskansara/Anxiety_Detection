import numpy as np
import argparse
import sys
import aria.sdk as aria
import wave
import io
import time
from queue import Queue
import threading
from projectaria_tools.core.sensor_data import AudioData, AudioDataRecord


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


class StreamingClientObserver:
    def __init__(self, audio_queue, output_file="audio_values.txt"):
        self.audio_queue = audio_queue
        self.audio_buffer = []
        self.buffer_duration = 10.0  # Desired buffer duration in seconds
        # self.sample_rate = SAMPLE_RATE
        self.output_file = output_file
        # Open the file in append mode so that it appends data instead of overwriting it
        self.file = open(self.output_file, "a")

    def __del__(self):
        # Ensure the file is closed when the object is destroyed
        self.file.close()

    def on_audio_received(self, audio_data: AudioData, record: AudioDataRecord):
        num_channels = 7
        audio_data = audio_data.data
        num_samples_per_channel = int(len(audio_data) / num_channels)
        audio_list = []

        for i in range(num_samples_per_channel):
            for c in range(num_channels):
                audio_val = audio_data[i * num_channels + c]
                audio_list.append(audio_val)
                self.file.write(f"{audio_val},")

        # Add a newline to separate each batch of audio values
        self.file.write("\n")


def is_valid_audio(audio):
    # Check if the array is a NumPy array
    if not isinstance(audio, np.ndarray):
        return False, "Audio data is not a NumPy array"

    # Check if the array dtype is int32 or float32
    if audio.dtype not in [np.int32, np.float32]:
        return False, f"Audio data type is {audio.dtype}, expected int32 or float32"

    # Check the dimensions of the array
    if audio.ndim not in [1, 2]:
        return False, f"Audio data has {audio.ndim} dimensions, expected 1 or 2"

    # Check for NaNs or Infs
    if np.isnan(audio).any():
        return False, "Audio data contains NaNs"
    if np.isinf(audio).any():
        return False, "Audio data contains Infs"

    # Check the amplitude range for int32
    if audio.dtype == np.int32:
        if audio.min() < -(2**31) or audio.max() > 2**31 - 1:
            return False, "Audio data values out of int32 range"

    # Check the amplitude range for float32
    if audio.dtype == np.float32:
        if audio.min() < -1.0 or audio.max() > 1.0:
            return False, "Audio data values out of float32 range"

    # If all checks pass, the audio is valid
    return True, "Audio data is valid"


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
        config.message_queue_size[aria.StreamingDataType.Audio] = 10
        config.security_options.use_ephemeral_certs = True
        streaming_client.subscription_config = config

        audio_queue = Queue()
        observer = StreamingClientObserver(audio_queue, output_file="audio_values.txt")
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
