import asyncio
import struct
from bfnet.Butterfly import AbstractButterfly
import msgpack

class PacketButterfly(AbstractButterfly):
    """
    A packeted Butterfly uses a Queue of Packets instead of
    a StreamReader/StreamWriter.
    """
    unpacker = struct.Struct("!2shh")

    def __init__(self, handler, loop: asyncio.AbstractEventLoop, max_packets=0):
        """
        Create a new Packeted Butterfly.
        """
        # First, super call it.
        super().__init__(handler, loop)

        # Create a new Packet queue.
        self.packet_queue = asyncio.Queue(loop=self._loop)

    @property
    def handler(self):
        return self._handler

    def data_received(self, data: bytes):
        """
        Parses out the Packet header, to create an appropriate new
        Packet object.
        :param data: The data to parse in.
        """
        self.logger.debug("Recieved new packet, deconstructing...")
        try:
            data = msgpack.unpackb(data, encoding="utf-8")
        except (msgpack.UnpackException, ValueError) as e:
            self.logger.error("Unable to unpack data: {}".format(e))
            self._transport.close()
            return
        if not 'id' in data:
            self.logger.error("ID not in packet data, skipping packet")
            self._transport.close()
            return
        self.logger.debug("New packet: ID {}, version {}".format(data["id"], data.get("version", "unknown")))
        try:
            packet_type = self.handler.packet_types[data['id']]
        except KeyError:
            self.logger.error("ID not valid for current handler.")
            self._transport.close()
            return
        packet = packet_type(self)
        if not 'data' in data:
            self.logger.error("No data field in packet, skipping data")
            self._transport.close()
            return
        else:
            created = packet.create(data['data'])
            if created:
                self._loop.create_task(self.packet_queue.put(packet))
            self.logger.debug("Created new packet")

    def connection_lost(self, exc):
        class FakeQueue(object):
            def get(self):
                return None

            @asyncio.coroutine
            def put(self):
                return

        self.packet_queue = FakeQueue()
        super().connection_lost(exc)

    def read(self):
        """
        Get a new packet off the Queue.
        """
        return (yield from self.packet_queue.get())

    def write(self, pack):
        """
        Write a packet to the client.
        :param pack: The packet to write. This will automatically add a header.
        """
        packed = msgpack.packb({"id": pack.id, "data": pack.gen()}, use_bin_type=True)
        self._transport.write(packed)
