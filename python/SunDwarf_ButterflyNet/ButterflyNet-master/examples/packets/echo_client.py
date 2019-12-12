import socket
import ssl
import struct
import msgpack

# Open a new SSL socket
size = 1024
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

s = ssl.wrap_socket(s)

s.connect(("127.0.0.1", 8001))

while True:
    to_send = input("> ")
    to_send_pak = msgpack.packb({"id": 2, "data": {"echo": "hello world!"}}, use_bin_type=True)

    s.write(to_send_pak)
    data = s.recv(size)
    print(msgpack.unpackb(data, encoding="UTF-8"))


s.close()
