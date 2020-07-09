from threading import Thread
import socket as skt
import random as rand
import librosa
import io
import numpy as np
import glob
import time

import warnings


def send_file(file_name):
    print("Sending File: "+file_name)
    stream_socket = skt.socket(skt.AF_INET, skt.SOCK_DGRAM)
    f, sr = librosa.load(original_data_path+file_name+".mp3")
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
            receive_socket.settimeout(0.1)
            data, _ = receive_socket.recvfrom(buf)
    except skt.timeout:
        receive_socket.close()
        waves = np.frombuffer(dest.getvalue(), dtype=np.float32)
        librosa.output.write_wav(
            packet_loss_data_path+file_name+".wav", waves, sr=22050)
        dest.close()
        print("Received File: "+file_name)


def get_file_name(idx):
    return f"sample-{idx:06}"


def stream_files(file_names):
    for file_name in file_names:
        thread = Thread(target=receive_file, args=(file_name, ))
        thread.start()

        send_file(file_name)
        thread.join()


if __name__ == "__main__":
    warnings.filterwarnings('ignore')
    host = "localhost"
    port = 9999
    buf = 1024

    addr = (host, port)

    original_data_path = "data/raw_data/"
    packet_loss_data_path = "data/packet_loss_data/"

    file_names = [f.split("\\")[1].split(".")[0]
                  for f in glob.glob("data/raw_data/*.mp3")]
    stream_files(file_names[:500])
