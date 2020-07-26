from threading import Thread
import socket as skt
import random as rand
import librosa
import io
import numpy as np
import glob
import time
import argparse
import warnings


def process_stft(stft):
    """Converts the complex values in the stft to real numbers in the shape [real, imaginary], makes time the primary axis, and pads time to a certain length"""
    assert MAX_LENGTH > stft.shape[1], f"Max length ({MAX_LENGTH}) is smaller then stft shape[1] ({stft.shape[1]})"
    # stft shape: (freq, frame, values)
    # After transpose: (frame, freq, values)
    transposed = np.transpose(np.dstack((stft.real, stft.imag)), (1, 0, 2))
    padded = np.pad(
        transposed, ((0, MAX_LENGTH - transposed.shape[0]), (0, 0), (0, 0)))
    real_stft = padded.reshape(
        padded.shape[0], padded.shape[1] * padded.shape[2])
    return real_stft


def save_stft(file_name, path_to_save, wave_data):
    """Saves the stft (with real values) to a file"""
    stft = librosa.stft(wave_data, n_fft=512, win_length=256)
    real_stft = process_stft(stft)

    np.save(path_to_save + file_name, real_stft)


def send_file(file_name):
    """Sends a file over the localhost with packet loss"""
    stream_socket = skt.socket(skt.AF_INET, skt.SOCK_DGRAM)
    f, sr = librosa.load(raw_data_path + file_name + ".mp3")
    save_stft(file_name, stft_original_path, f)
    byte_stream = io.BytesIO(f.tobytes())
    data = byte_stream.read(buf)
    packet_losses = False
    while (data):
        if(rand.random() < 0.05):
            packet_losses = True
        if(packet_losses):
            stream_socket.sendto(bytearray(len(data)), addr)
            if(rand.random() < 0.2):
                packet_losses = False
        else:
            stream_socket.sendto(data, addr)
        data = byte_stream.read(buf)

    byte_stream.close()
    stream_socket.close()


def receive_file(file_name):
    """Receives a file with packet loss over localhost and saves it"""
    dest = io.BytesIO()
    receive_socket = skt.socket(skt.AF_INET, skt.SOCK_DGRAM)

    receive_socket.bind(addr)
    data, _ = receive_socket.recvfrom(buf)
    try:
        while(data):
            dest.write(data)
            receive_socket.settimeout(0.05)
            data, _ = receive_socket.recvfrom(buf)
    except skt.timeout:
        receive_socket.close()
        wave_data = np.frombuffer(dest.getvalue(), dtype=np.float32)
        save_thread = Thread(target=save_stft, args=(
            file_name, stft_packet_loss_path, wave_data))
        save_thread.start()
        # save_stft(file_name, stft_packet_loss_path, wave_data)
        librosa.output.write_wav(
            packet_loss_data_path + file_name + ".wav", wave_data, sr=22050)
        dest.close()


def get_file_name(idx):
    return f"sample-{idx:06}"


def stream_files(file_names):
    """Automates sending/receiving files."""
    for i, file_name in enumerate(file_names):
        start_time = time.time()
        thread = Thread(target=receive_file, args=(file_name, ))
        thread.start()

        send_file(file_name)
        thread.join()
        end_time = time.time() - start_time

        print(f"Processed: {file_name}, Time: {end_time:.2f}s", end="\r")
        if(i == amount - 1):
            break
    print()
    print("Done!")


if __name__ == "__main__":
    warnings.filterwarnings('ignore')

    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--amount", type=int)
    args = parser.parse_args()

    amount = args.amount
    MAX_LENGTH = 5000
    host = "localhost"
    port = 9999
    buf = 1024

    addr = (host, port)

    raw_data_path = "data/sample_raw/"
    packet_loss_data_path = "data/sample_loss/"
    stft_original_path = "data/stft_original/"
    stft_packet_loss_path = "data/stft_loss/"

    file_names = [f.split("\\")[1].split(".")[0]
                  for f in glob.glob("data/sample_raw/*.mp3")]
    print("Creating packet-loss data...")
    stream_files(file_names[:500])
