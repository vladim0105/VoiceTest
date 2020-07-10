from threading import Thread
import socket as skt
import random as rand
import librosa
import io
import numpy as np
import glob
import time

import warnings


def stft_to_real(stft):
    real_stft = np.zeros(
        (stft.shape[0], stft.shape[1], 2))  # pylint: disable=E1136

    for idx, x in np.ndenumerate(stft):
        real_stft[idx[0], idx[1], 0] = x.real
        real_stft[idx[0], idx[1], 1] = x.imag
    return real_stft


def save_stft(file_name, path_to_save, wave_data):
    stft = librosa.stft(wave_data, n_fft=512, win_length=256)
    real_stft = stft_to_real(stft)
    np.save(path_to_save+file_name, real_stft)


def send_file(file_name):
    stream_socket = skt.socket(skt.AF_INET, skt.SOCK_DGRAM)
    f, sr = librosa.load(raw_data_path+file_name+".mp3")
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
        save_stft(file_name, stft_packet_loss_path, wave_data)
        librosa.output.write_wav(
            packet_loss_data_path+file_name+".wav", wave_data, sr=22050)
        dest.close()


def get_file_name(idx):
    return f"sample-{idx:06}"


def stream_files(file_names):
    for file_name in file_names:
        start_time = time.time()
        thread = Thread(target=receive_file, args=(file_name, ))
        thread.start()

        send_file(file_name)
        thread.join()
        end_time = time.time()-start_time

        print(f"Processed: {file_name}, Time: {end_time:.2f}s", end="\r")
    print()
    print("Done!")


if __name__ == "__main__":
    warnings.filterwarnings('ignore')
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
