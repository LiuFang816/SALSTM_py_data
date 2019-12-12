import logging
import ssl
import string
from bfnet import Net
from bfnet.Butterfly import Butterfly
from bfnet.BFHandler import ButterflyHandler
import asyncio


logging.basicConfig(filename='/dev/null', level=logging.INFO)

formatter = logging.Formatter('%(asctime)s - [%(levelname)s] %(name)s - %(message)s')
root = logging.getLogger()

consoleHandler = logging.StreamHandler()
consoleHandler.setFormatter(formatter)
root.addHandler(consoleHandler)

# Create your event loop.
loop = asyncio.get_event_loop()


class MyHandler(ButterflyHandler):
    @asyncio.coroutine
    def on_connection(self, butterfly: Butterfly):
        # Here we re-purpose the Butterfly dict.
        # Read in their nickname.
        butterfly.write(b"Nickname: ")
        nick = yield from butterfly.read()
        nick = nick.rstrip(b'\n').rstrip(b'\r')
        # Tell the others somebody has connected.
        self.logger.debug("{} has joined".format(nick.decode()))
        for bf, _ in self.butterflies.values():
            bf.write(nick + b" has joined the room\n")
        # Set the `nick` attribute on the Butterfly.
        butterfly.nick = nick
        # Begin handling normally.
        fut = self.begin_handling(butterfly)
        self.butterflies[nick] = (butterfly, fut)


    @asyncio.coroutine
    def on_disconnect(self, butterfly: Butterfly):
        # Override the on_disconnect to cancel the correct futures.
        if not hasattr(butterfly, "nick"):
            self.logger.warning("Connection cancelled before on_connect finished - will be killed soon!")
            return
        if butterfly.nick in self.butterflies:
            bf = self.butterflies.pop(butterfly.nick)
            assert isinstance(bf, tuple)
            assert len(bf) == 2
            bf[1].cancel()
        for bf, _ in self.butterflies.values():
            bf.write(butterfly.nick + b" has left the room.\n")


@asyncio.coroutine
def main():
    my_handler = MyHandler.get_handler(loop=loop, log_level=logging.DEBUG)
    my_server = yield from my_handler.create_server(("127.0.0.1", 8001), ("localhost.crt", "server.key", None))


    # Define our simple coroutine for handling messages.
    @my_server.any_data
    @asyncio.coroutine
    def _handle_data(data: bytes, butterfly: Butterfly, handler: ButterflyHandler):
        # Echo messages to all other Butterflies.
        for bf in handler.butterflies.values():
            assert isinstance(bf, tuple), "bf should be a tuple -> {}".format(bf)
            if bf[0] != butterfly:
                bf[0].write(butterfly.nick + b": " + data)


if __name__ == '__main__':
    loop.create_task(main())
    try:
        loop.run_forever()
    except KeyboardInterrupt:
        exit(0)
