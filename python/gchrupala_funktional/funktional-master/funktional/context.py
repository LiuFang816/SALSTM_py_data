import sys
from contextlib import contextmanager

# Are we training (or testing)
training = False


@contextmanager
def context(**kwargs):
    """Temporarily change the values of context variables passed.

    Enables the `with` syntax:

    >>> with context(training=True):
    ...  
    """
    current = dict((k, getattr(sys.modules[__name__], k)) for k in kwargs)
    for k,v in kwargs.items():
        setattr(sys.modules[__name__], k, v)
    yield
    for k,v in current.items():
        setattr(sys.modules[__name__], k, v)
        
