import socket as skt
import random as rand
s = skt.socket(skt.AF_INET, skt.SOCK_DGRAM)
host = "localhost"
port = 9999
buf = 1024

addr = (host, port)

file_name = "sample.opus"


f = open(file_name, "rb")
data = f.read(buf)

while (data):
    if(s.sendto(data, addr)):
        if(rand.random() < 0.1):
            print("lagg...")
            continue
        print("sending ...")
        data = f.read(buf)
s.close()
f.close()
