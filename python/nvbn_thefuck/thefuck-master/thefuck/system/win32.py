import os
import sys
import msvcrt
import win_unicode_console
from .. import const


def init_output():
    import colorama
    win_unicode_console.enable()
    colorama.init()


def get_key():
    ch = msvcrt.getch()
    if ch in (b'\x00', b'\xe0'):  # arrow or function key prefix?
        ch = msvcrt.getch()  # second call returns the actual key code

    if ch == b'\x03':
        return const.KEY_CTRL_C
    if ch == b'H':
        return const.KEY_UP
    if ch == b'P':
        return const.KEY_DOWN

    encoding = sys.stdout.encoding or os.environ.get('PYTHONIOENCODING', 'utf-8')
    return ch.decode(encoding)

try:
    from pathlib import Path
except ImportError:
    from pathlib2 import Path


def _expanduser(self):
    return self.__class__(os.path.expanduser(str(self)))

# pathlib's expanduser fails on windows, see http://bugs.python.org/issue19776
Path.expanduser = _expanduser
