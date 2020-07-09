import select
import sys
from socket import *

host = "localhost"
port = 9999
s = socket(AF_INET, SOCK_DGRAM)
s.bind((host, port))

addr = (host, port)
buf = 1024

dest = open("dest.opus", "wb")

data, addr = s.recvfrom(buf)
try:
    while(data):
        dest.write(data)
        s.settimeout(2)
        data, addr = s.recvfrom(buf)
except timeout:
    s.close()
    dest.close()
    print("File Downloaded")
