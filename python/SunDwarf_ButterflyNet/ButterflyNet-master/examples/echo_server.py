import asyncio
import logging
import ssl


logging.basicConfig(filename='/dev/null', level=logging.INFO)

formatter = logging.Formatter('%(asctime)s - [%(levelname)s] %(name)s - %(message)s')
root = logging.getLogger()

consoleHandler = logging.StreamHandler()
consoleHandler.setFormatter(formatter)
root.addHandler(consoleHandler)

import bfnet
from bfnet import Net, Butterfly


# Create your event loop.
loop = asyncio.get_event_loop()

my_handler = bfnet.get_handler(loop, log_level=logging.DEBUG, buffer_size=4096)


@asyncio.coroutine
def main():
    my_server = yield from my_handler.\
        create_server(("127.0.0.1", 8001), ("keys/test.crt", "keys/test.key", None))

    @my_server.any_data
    @asyncio.coroutine
    def echo(data: bytes, butterfly: Butterfly, handler: bfnet.ButterflyHandler):
        butterfly.write(data)


if __name__ == '__main__':
    loop.create_task(main())
    try:
        loop.run_forever()
    except KeyboardInterrupt:
        # Close the server.
        my_handler.stop()
        loop.stop()

