import asyncio
import logging
import ssl

from bfnet.BFHandler import ButterflyHandler
from bfnet.packets.PacketButterfly import PacketButterfly
from .Packets import BasePacket
from .PacketNet import PacketNet


class PacketHandler(ButterflyHandler):
    """
    A PacketHandler is a type of Handler that allows you to send/recieve Packets instead
    of using raw data.

    It is a step above the normal ButterflyNet bytes layer, by instead using structures
    for easy OO-style networking information.
    """

    def __init__(self, event_loop: asyncio.AbstractEventLoop, ssl_context: ssl.SSLContext=None,
            loglevel: int=logging.DEBUG, buffer_size: int=asyncio.streams._DEFAULT_LIMIT):
        super().__init__(event_loop, ssl_context, loglevel, 0)

        # Define a new dict of Packet types.
        self.packet_types = {}
        # Define the basic packet type.
        self.basic_packet_type = BasePacket

        # Define the default Net type.
        self.default_net = PacketNet

    def butterfly_factory(self):
        """
        Creates a new PacketedButterfly instead of a normal Butterfly.
        :return: A new :class:`bfnet.packets.PacketButterfly`.
        """
        return PacketButterfly(self, self._event_loop)

    def add_packet_type(self, pack: BasePacket):
        """
        Adds a new Packet type to your handler.

        This can be used as a decorator or a normal method, as it returns the class.

        Do NOT use on an instance of a class.
        :param pack: The packet to use.
        :return: Your packet class back.
        """
        # Get the ID.
        id = pack.id
        self.packet_types[id] = pack

        return pack
