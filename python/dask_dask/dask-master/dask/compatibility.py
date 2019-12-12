# flake8: noqa
from __future__ import absolute_import, division, print_function

import functools
import inspect
import operator
import sys
import types

PY3 = sys.version_info[0] == 3
PY2 = sys.version_info[0] == 2

class LZMAFile:
    def __init__(self, *args, **kwargs):
        raise ValueError("xz files requires the lzma module. "
                            "To use, install lzmaffi or backports.lzma.")
LZMA_AVAILABLE = False

if PY3:
    import builtins
    from queue import Queue, Empty
    from itertools import zip_longest
    from io import StringIO, BytesIO
    from bz2 import BZ2File
    from gzip import (GzipFile, compress as gzip_compress,
            decompress as gzip_decompress)
    try:
        try:
            from lzmaffi import (LZMAFile, compress as lzma_compress,
                                 decompress as lzma_decompress)
        except ImportError:
            from lzma import (LZMAFile, compress as lzma_compress,
                              decompress as lzma_decompress)
        LZMA_AVAILABLE = True
    except ImportError:
        # Fallback to top-level definition
        pass

    from urllib.request import urlopen
    from urllib.parse import urlparse, urlsplit, quote, unquote
    FileNotFoundError = FileNotFoundError
    unicode = str
    long = int
    zip = zip
    def apply(func, args, kwargs=None):
        if kwargs:
            return func(*args, **kwargs)
        else:
            return func(*args)
    range = range
    reduce = functools.reduce
    operator_div = operator.truediv

    def _getargspec(func):
        return inspect.getfullargspec(func)

else:
    import __builtin__ as builtins
    from Queue import Queue, Empty
    from itertools import izip_longest as zip_longest, izip as zip
    from StringIO import StringIO
    from io import BytesIO, BufferedIOBase
    import bz2
    import gzip
    from urllib2 import urlopen
    from urlparse import urlparse, urlsplit
    from urllib import quote, unquote
    unicode = unicode
    long = long
    apply = apply
    range = xrange
    reduce = reduce
    operator_div = operator.div
    FileNotFoundError = IOError

    def _getargspec(func):
        return inspect.getargspec(func)

    def gzip_decompress(b):
        f = gzip.GzipFile(fileobj=BytesIO(b))
        result = f.read()
        f.close()
        return result

    def gzip_compress(b):
        bio = BytesIO()
        f = gzip.GzipFile(fileobj=bio, mode='w')
        f.write(b)
        f.close()
        bio.seek(0)
        result = bio.read()
        return result

    if sys.version_info[1] <= 7:
        class BZ2File(BufferedIOBase):
            def __init__(self, *args, **kwargs):
                self.__obj = bz2.BZ2File(*args, **kwargs)

            def close(self):
                return self.__obj.close()

            @property
            def closed(self):
                return self.__obj.closed

            def flush(self):
                pass

            def isatty(self):
                return self.__obj.isatty()

            def read(self, *args, **kwargs):
                return self.__obj.read(*args, **kwargs)

            def read1(self, *args, **kwargs):
                return self.__obj.read(*args, **kwargs)

            def readable(self):
                return 'r' in self.__obj.mode

            def readline(self, *args, **kwargs):
                return self.__obj.readline(*args, **kwargs)

            def readlines(self, *args, **kwargs):
                return self.__obj.readlines(*args, **kwargs)

            def seek(self, *args, **kwargs):
                self.__obj.seek(*args, **kwargs)
                return self.tell()

            def seekable(self):
                return self.readable()

            def tell(self):
                return self.__obj.tell()

            def truncate(self, *args, **kwargs):
                return self.__obj.truncate(*args, **kwargs)

            def writable(self):
                return 'w' in self.__obj.mode

            def write(self, *args, **kwargs):
                return self.__obj.write(*args, **kwargs)

            def writelines(self, *args, **kwargs):
                return self.__obj.writelines(*args, **kwargs)
    else:
        BZ2File = bz2.BZ2File

    class GzipFile(BufferedIOBase):
        def __init__(self, *args, **kwargs):
            self.__obj = gzip.GzipFile(*args, **kwargs)

        def close(self):
            return self.__obj.close()

        @property
        def closed(self):
            return self.__obj.fileobj is None

        def flush(self, *args, **kwargs):
            return self.__obj.flush(*args, **kwargs)

        def isatty(self):
            return self.__obj.isatty()

        def read(self, *args, **kwargs):
            return self.__obj.read(*args, **kwargs)

        def read1(self, *args, **kwargs):
            return self.__obj.read(*args, **kwargs)

        def readable(self):
            return self.__obj.mode == gzip.READ

        def readline(self, *args, **kwargs):
            return self.__obj.readline(*args, **kwargs)

        def readlines(self, *args, **kwargs):
            return self.__obj.readlines(*args, **kwargs)

        def seek(self, *args, **kwargs):
            self.__obj.seek(*args, **kwargs)
            return self.tell()

        def seekable(self):
            # See https://hg.python.org/cpython/file/2.7/Lib/gzip.py#l421
            return True

        def tell(self):
            return self.__obj.tell()

        def truncate(self, *args, **kwargs):
            return self.__obj.truncate(*args, **kwargs)

        def writable(self):
            return self.__obj.mode == gzip.WRITE

        def write(self, *args, **kwargs):
            return self.__obj.write(*args, **kwargs)

        def writelines(self, *args, **kwargs):
            return self.__obj.writelines(*args, **kwargs)

    try:
        try:
            from lzmaffi import (LZMAFile, compress as lzma_compress,
                                 decompress as lzma_decompress)
        except ImportError:
            from backports.lzma import LZMAFile
            from backports.lzma import (LZMAFile, compress as lzma_compress,
                                        decompress as lzma_decompress)
        LZMA_AVAILABLE = True
    except ImportError:
        # Fallback to top-level definition
        pass


def getargspec(func):
    """Version of inspect.getargspec that works for functools.partial objects"""
    if isinstance(func, functools.partial):
        return _getargspec(func.func)
    else:
        if isinstance(func, type):
            return _getargspec(func.__init__)
        else:
            return _getargspec(func)

def skip(func):
    return


def bind_method(cls, name, func):
    """Bind a method to class

    Parameters
    ----------

    cls : type
        class to receive bound method
    name : basestring
        name of method on class instance
    func : function
        function to be bound as method

    Returns
    -------
    None
    """
    # only python 2 has bound/unbound method issue
    if not PY3:
        setattr(cls, name, types.MethodType(func, None, cls))
    else:
        setattr(cls, name, func)
