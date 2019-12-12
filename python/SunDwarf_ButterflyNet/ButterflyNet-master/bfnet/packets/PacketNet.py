import types

from bfnet.Net import Net


class PacketNet(Net):
    """
    A PacketNet is a special type of Net that works on Packets
    instead of data.
    """
    def __init__(self, ip, port, loop, server):
        super().__init__(ip, port, loop, server)
        # Set the real handler.
        self._real_handler = None

    def handle(self, butterfly):
        """
        Stub method that calls your REAL handler.

        This would normally be a coroutine, but a task will be created from
        the returned handler.
        """
        try:
            return self._real_handler(butterfly)
        except TypeError as e:
            raise TypeError("Packet handler has not been set.") from e

    def set_handler(self, func: types.GeneratorType):
        """
        Set the default Packet handler.

        This can be used as a decorator, or as a normal call.

        The handler MUST be a coroutine.

        This handler MUST be an infinite loop. Failure to do so will mean your packets will stop being
        handled after one packet arrives.
        :param func: The function to set as the handler.
        :return: Your function back.
        """
        self._real_handler = func
        return func
