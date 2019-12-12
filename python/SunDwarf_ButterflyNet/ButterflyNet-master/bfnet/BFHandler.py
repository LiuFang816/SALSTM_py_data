import _ssl
import asyncio
import logging
import multiprocessing
import ssl
import sys
import types
from concurrent import futures
from bfnet.Butterfly import Butterfly
from bfnet.Net import Net


class ButterflyHandler(object):
    """
    A ButterflyHandler is a class that describes what happens when a Butterfly is caught by a net.

    It has several methods that are automatically called at critical stages in the connection:
        - :func:`ButterflyHandler.on_connection`
        - :func:`ButterflyHandler.on_disconnect`

    These methods are called at the appropriate time, as their name describes.
    """
    instance = None

    def __init__(self, event_loop: asyncio.AbstractEventLoop, ssl_context: ssl.SSLContext=None,
            loglevel: int=logging.DEBUG, buffer_size: int=asyncio.streams._DEFAULT_LIMIT):
        """
        Create a new ButterflyHandler.

        This class should not be called directly. Instead, use ButterflyHandler.get_handler() to
        get a reference instead.

        :param event_loop: The :class:`asyncio.BaseEventLoop` to use for the server.
        :param ssl_context: The :class:`ssl.SSLContext` to use for the server.
        :param loglevel: The logging level to use.
        :param buffer_size: The buffer size to use.
        """
        self._event_loop = event_loop
        self._server = None
        if not ssl_context:
            # This looks very similar to the code for create_default_context
            # That's because it is the code
            # For some reason, create_default_context doesn't like me and won't work properly
            self._ssl = ssl.SSLContext(protocol=ssl.PROTOCOL_SSLv23)
            # SSLv2 considered harmful.
            self._ssl.options |= ssl.OP_NO_SSLv2

            # SSLv3 has problematic security and is only required for really old
            # clients such as IE6 on Windows XP
            self._ssl.options |= ssl.OP_NO_SSLv3
            self._ssl.load_default_certs(ssl.Purpose.SERVER_AUTH)
            self._ssl.options |= getattr(_ssl, "OP_NO_COMPRESSION", 0)
            self._ssl.set_ciphers(ssl._RESTRICTED_SERVER_CIPHERS)
            self._ssl.options |= getattr(_ssl, "OP_CIPHER_SERVER_PREFERENCE", 0)

        else:
            self._ssl = ssl_context

        self._bufsize = buffer_size
        self.default_butterfly = Butterfly
        self.default_net = Net

        self._executor = futures.ThreadPoolExecutor(max_workers=multiprocessing.cpu_count() * 2 + 1)

        self.net = None
        self.log_level = loglevel
        self.logger = logging.getLogger("ButterflyNet")
        self.logger.setLevel(loglevel)
        if self.logger.level <= logging.DEBUG:
            self._event_loop.set_debug(True)

        self.butterflies = {}

    def stop(self):
        """
        Stop a Net.

        This will kill all handlers, disconnect all butterflies, and unbind the server.
        """
        self.logger.info("Stopping server.")
        print("Stopping server.")
        # Loop over our Butterflies.
        for _, bf in self.butterflies.items():
            assert isinstance(bf, tuple), "bf should be a tuple (bf, fut) -> {}".format(bf)
            # Cancel the future.
            bf[1].cancel()
            # Cancel the Butterfly.
            bf[0].stop()
        try:
            self.net.stop()
        except AttributeError:
            pass
        self._event_loop.stop()

    @asyncio.coroutine
    def on_connection(self, butterfly: Butterfly):
        """
        Stub for an on_connection event.

        This will call the data handler, and save the result.

        This method is a coroutine.
        :param butterfly: The butterfly object created.
        """
        # Begin handling.
        handler = self.begin_handling(butterfly)
        # Create a new entry in our butterfly table.
        self.butterflies["{}:{}".format(butterfly.ip, butterfly.client_port)] = (butterfly, handler)

    @asyncio.coroutine
    def on_disconnect(self, butterfly: Butterfly):
        """
        Stub for an on_disconnect event.

        This will kill the data handler.

        This method is a coroutine.
        :param butterfly: The butterfly object created.
        """
        s = "{}:{}".format(butterfly.ip, butterfly.client_port)
        if s in self.butterflies:
            bf = self.butterflies.pop(s)
            # These are here by default - don't call super() if you modify the butterfly dict!
            assert isinstance(bf, tuple)
            assert len(bf) == 2
            bf[1].cancel()

    def begin_handling(self, butterfly: Butterfly):
        """
        Begin the handler loop and start handling data that flows in.

        This will schedule the Net's handle() coroutine to run soon.
        :return A Future object for the handle() coroutine.
        """
        res = self.net.handle(butterfly)
        return self._event_loop.create_task(res)

    def async_func(self, fun: types.FunctionType) -> asyncio.Future:
        """
        Turns a blocking function into an async function by running it inside an executor.

        This executor is by default a :class:`~concurrent.futures.ThreadPoolExecutor`.
        :param fun: The function to run async.
            If you wish to pass parameters to this func, use
            functools.partial (https://docs.python.org/3/library/functools.html#functools.partial).
        :return: A :class:`~asyncio.Future` object for the function.
        """
        future = self._event_loop.run_in_executor(self._executor, fun)
        return future

    def async_and_wait(self, fun: types.FunctionType):
        """
        Turns a blocking function into an async function by running it inside an executor. It then uses
        :func:`~asyncio.wait_for` to wait for the Future to complete.

        This executor is by default a :class:`~concurrent.futures.ThreadPoolExecutor`.
        :param fun: The function to run async.
            If you wish to pass parameters to this func, use
            functools.partial (https://docs.python.org/3/library/functools.html#functools.partial).
        :return: The result of the function.
        """
        future = self.async_func(fun)
        return (yield from asyncio.wait_for(future, loop=self._event_loop))

    def create_task(self, coro: types.FunctionType):
        """
        Create a new task on the event loop, and return the :class:`~asyncio.Future` created.
        :param coro: A coroutine or future to add.
        :return: The Future created.
        """
        future = self._event_loop.create_task(coro)
        return future

    def call_soon(self, coro: types.FunctionType, *args):
        """
        Call a coroutine or Future as soon as possible on the event loop.
        :param coro: The coroutine or Future to call.
        :param args: The arguments to the callback.
        """
        handle = self._event_loop.call_soon(coro, args)
        return handle

    def set_executor(self, executor: futures.Executor):
        """
        Set the default executor for use with async_func.
        :param executor: A :class:`~concurrent.futures.Executor` to set as the executor.
        """
        self._executor = executor

    def _load_ssl(self, ssl_options: tuple):
        """
        Internal call used to load SSL parameters from the SSL option tuple.

        Do not touch.
        :param ssl_options: The SSL options to use.
        """
        try:
            self._ssl.load_cert_chain(certfile=ssl_options[0], keyfile=ssl_options[1], password=ssl_options[2])
        except IOError as e:
            self.logger.error("Unable to load certificate files: {}".format(e))
            self.stop()

    @classmethod
    def get_handler(cls, loop: asyncio.AbstractEventLoop, ssl_context: ssl.SSLContext=None,
            log_level: int=logging.INFO, buffer_size: int=asyncio.streams._DEFAULT_LIMIT):
        """
        Get the instance of the handler currently running.

        :param loop: The :class:`asyncio.BaseEventLoop` to use for the server.
        :param ssl_context: The :class:`ssl.SSLContext` to use for the server.
        :param log_level: The logging level to use.
        :param buffer_size: The buffer size to use.
        """
        if not cls.instance:
            cls.instance = cls(loop, ssl_context, log_level, buffer_size)
        return cls.instance

    def butterfly_factory(self):
        """
        Create a new :class:`Butterfly` instance.

        This method will create a new Butterfly from the default Butterfly
        creator specified by self.default_butterfly.

        Override this if you use a different constructor in your Butterfly.
        :return:
        """
        bf = self.default_butterfly(loop=self._event_loop, bufsize=self._bufsize, handler=self)
        return bf

    @asyncio.coroutine
    def create_server(self, bind_options: tuple, ssl_options: tuple) -> Net:
        """
        Create a new server using the event loop specified.

        This method is a coroutine.
        :param bind_options: The IP and port to bind to on the server.
        :param ssl_options: A tuple of SSL options:
            - The certificate file to use
            - The private key to use
            - The private key password, or None if it does not have a password.
        :return: A :class:`bfnet.Net.Net` object.
        """

        # Load SSL.
        self._load_ssl(ssl_options)

        # Create the server.
        host, port = bind_options
        self._server = yield from self._event_loop.create_server(self.butterfly_factory, host=host, port=port,
            ssl=self._ssl)
        # Create the Net.
        # Use the default net.
        self.net = self.default_net(ip=host, port=port, loop=self._event_loop, server=self._server)
        self.net._set_bf_handler(self)
        # Create a signal handler.
        if sys.platform != "win32":
            self._event_loop.add_signal_handler(15, self.stop)
        return self.net


ButterflyHandler.get_handler.__annotations__['return'] = ButterflyHandler
