import asyncio
import logging
import struct

from bfnet.packets import PacketHandler, Packet, PacketButterfly
from bfnet import util


logging.basicConfig(filename='/dev/null', level=logging.INFO)

formatter = logging.Formatter('%(asctime)s - [%(levelname)s] %(name)s - %(message)s')
root = logging.getLogger()

consoleHandler = logging.StreamHandler()
consoleHandler.setFormatter(formatter)
root.addHandler(consoleHandler)

# Create your event loop.
loop = asyncio.get_event_loop()

my_handler = PacketHandler.get_handler(loop=loop, log_level=logging.DEBUG)


# Create a new packet.
@my_handler.add_packet_type
class Packet0Echo(Packet):
    id = 0


    def __init__(self, pbf):
        super().__init__(pbf)
        # Set our attributes.
        self.data_to_echo = ""


    def unpack(self, data: dict):
        """
        Unpack the packet.
        """
        self.data_to_echo = data["echo"]
        return True


    def gen(self):
        """
        Pack a new packet.
        """
        return {"echo": self.data_to_echo}


@asyncio.coroutine
def main():
    my_server = yield from my_handler.create_server(("127.0.0.1", 8001), ("keys/test.crt", "keys/test.key", None))


    @my_server.set_handler
    @asyncio.coroutine
    def handler(bf: PacketButterfly):
        while True:
            echopacket = yield from bf.read()
            if not echopacket:
                break
            bf.write(echopacket)


if __name__ == '__main__':
    loop.create_task(main())
    try:
        loop.run_forever()
    except KeyboardInterrupt:
        # Close the server.
        my_handler.stop()
        loop.close()
