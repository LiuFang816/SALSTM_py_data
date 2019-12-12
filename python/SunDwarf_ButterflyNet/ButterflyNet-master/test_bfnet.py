import socket
import sys
import time
import subprocess
import ssl


def test_server_listening():
    # Check if it's listening.
    example_server = subprocess.Popen("{} examples/echo_server.py".format(sys.executable).split())
    time.sleep(1)
    assert subprocess.check_call("nc -z localhost 8001".split()) == 0
    example_server.kill()


def test_server_echo():
    example_server = subprocess.Popen("{} examples/echo_server.py".format(sys.executable).split())
    time.sleep(1)
    assert subprocess.check_call("nc -z localhost 8001".split()) == 0
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s = ssl.wrap_socket(s)

    s.connect(("127.0.0.1", 8001))

    s.write(b"HELLO")
    data = s.recv()
    assert data
    assert data == b"HELLO"

    example_server.kill()


