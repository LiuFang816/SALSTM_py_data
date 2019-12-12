import asyncio
import logging


class AbstractButterfly(asyncio.Protocol):
    """
    A butterfly represents a client object that has connected to your server.

    An abstract butterfly is an abstract class that only contains a minimal amount
    of default information.
    """

    def __init__(self, handler, loop: asyncio.AbstractEventLoop):
        """
        Create a new Butterfly.

        :param handler: The :class:`ButterflyHandler` to set as our handler.
        :param loop: The :class:`asyncio.BaseEventLoop` to use.
        """
        self._loop = loop
        self._handler = handler

        self._transport = None

        self.ip = "0.0.0.0"
        self.port = 0

        self.logger = logging.getLogger("ButterflyNet")
        self.logger.setLevel(self._handler.log_level)

    def connection_made(self, transport: asyncio.Transport):
        """
        Called upon a connection being made.
        :param transport: The transport created.
        """
        super().connection_made(transport)
        self._transport = transport
        self.ip, self.client_port = transport.get_extra_info("peername")
        self.logger.info("Recieved connection from {}:{}".format(*transport.get_extra_info("peername")))

        # Call our handler.
        res = self._handler.on_connection(self)

        if asyncio.coroutines.iscoroutine(res):
            self._loop.create_task(res)

    def connection_lost(self, exc):
        """
        Called upon a connection being lost.
        :param exc: The exception data to use.
        """
        super().connection_lost(exc)
        self.logger.info("Lost connection from {}:{}".format(self.ip, self.client_port))

        # Call the handler.
        res = self._handler.on_disconnect(self)
        if asyncio.coroutines.iscoroutine(res):
            self._loop.create_task(res)

    def stop(self):
        """
        Kills the Butterfly.
        """
        self._transport.close()

    def read(self) -> bytes:
        """
        Read all available data from the Butterfly.
        """
        pass

    @asyncio.coroutine
    def drain(self):
        """
        Drain the writer, if applicable.
        """
        pass

    def write(self, data: bytes):
        """
        Write to the butterfly.
        :param data: The byte data to write.
        """
        pass


class Butterfly(AbstractButterfly):
    """
    A butterfly represents a client object that has connected to your server.

    This will automatically call the appropriate methods in your handler, and set information about ourselves.
    """

    def __init__(self, handler, bufsize, loop: asyncio.AbstractEventLoop):
        """
        Create a new butterfly.
        :param handler: The :class:`ButterflyHandler` to set as our handler.
        :param bufsize: The buffersize to use for reading.
        :param loop: The :class:`asyncio.BaseEventLoop` to use.
        """
        super().__init__(handler, loop=loop)
        self._streamreader = asyncio.StreamReader(loop=self._loop)
        self._streamwriter = None

        self._bufsize = bufsize

    def connection_made(self, transport: asyncio.Transport):
        """
        Called upon a connection being made.

        This will automatically call your BFHandler.on_connection().
        :param transport: The transport to set the streamreader/streamwriter to.
        """
        self._streamreader.set_transport(transport)
        self._streamwriter = asyncio.StreamWriter(transport, self, self._streamreader, self._loop)
        super().connection_made(transport)

    def connection_lost(self, exc):
        """
        Called upon a connection being lost.

        This will automatically call your BFHandler.on_disconnect().
        :param exc: The exception data from asyncio.
        """
        if exc is None:
            self._streamreader.feed_eof()
        else:
            self._streamreader.set_exception(exc)
        super().connection_lost(exc)

    def data_received(self, data: bytes):
        """
        Called upon data received.

        This will only automatically call your Net.handle() method IF
        your `should_handle` attribute on the butterfly is True.

        Otherwise, it will simply pass your data into the StreamReader.
        :param data: The data to handle.
        """
        self.logger.debug("Recieved data: {}".format(data))
        self._streamreader.feed_data(data)

    def eof_received(self):
        """
        Called upon EOF recieved.
        """
        self._streamreader.feed_eof()
        return True

    @asyncio.coroutine
    def read(self) -> bytes:
        """
        Read in data from the Butterfly.
        :return: Bytes containing data from the butterfly.
        """
        return (yield from self._streamreader.read(self._bufsize))

    @asyncio.coroutine
    def drain(self):
        """
        Drain the writer.
        """
        return (yield from self._streamwriter.drain())

    def write(self, data: bytes):
        """
        Write to the butterfly.
        :param data: The byte data to write.
        """
        self._streamwriter.write(data)
