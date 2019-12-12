import multiprocessing
import numpy as np
import pickle
from operator import add

from dask.context import set_options
from dask.multiprocessing import get, _dumps, _loads
from dask.utils_test import inc


def test_pickle_globals():
    """ For the function f(x) defined below, the only globals added in pickling
    should be 'np' and '__builtins__'"""
    def f(x):
        return np.sin(x) + np.cos(x)

    assert set(['np', '__builtins__']) == set(
        _loads(_dumps(f)).__globals__.keys())


def bad():
    raise ValueError("12345")


def test_errors_propagate():
    dsk = {'x': (bad,)}

    try:
        get(dsk, 'x')
    except Exception as e:
        assert isinstance(e, ValueError)
        assert "12345" in str(e)


def make_bad_result():
    return lambda x: x + 1


def test_unpicklable_results_generate_errors():

    dsk = {'x': (make_bad_result,)}

    try:
        get(dsk, 'x')
    except Exception as e:
        # can't use type because pickle / cPickle distinction
        assert type(e).__name__ in ('PicklingError', 'AttributeError')


class NotUnpickleable(object):
    def __getstate__(self):
        return ()

    def __setstate__(self, state):
        raise ValueError("Can't unpickle me")


def test_unpicklable_args_generate_errors():
    a = NotUnpickleable()

    def foo(a):
        return 1

    dsk = {'x': (foo, a)}

    try:
        get(dsk, 'x')
    except Exception as e:
        assert isinstance(e, ValueError)

    dsk = {'x': (foo, 'a'),
           'a': a}

    try:
        get(dsk, 'x')
    except Exception as e:
        assert isinstance(e, ValueError)


def test_reuse_pool():
    pool = multiprocessing.Pool()
    with set_options(pool=pool):
        assert get({'x': (inc, 1)}, 'x') == 2
        assert get({'x': (inc, 1)}, 'x') == 2


def test_dumps_loads():
    with set_options(func_dumps=pickle.dumps, func_loads=pickle.loads):
        assert get({'x': 1, 'y': (add, 'x', 2)}, 'y') == 3


def test_fuse_doesnt_clobber_intermediates():
    d = {'x': 1, 'y': (inc, 'x'), 'z': (add, 10, 'y')}
    assert get(d, ['y', 'z']) == (2, 12)


def test_optimize_graph_false():
    from dask.callbacks import Callback
    d = {'x': 1, 'y': (inc, 'x'), 'z': (add, 10, 'y')}
    keys = []
    with Callback(pretask=lambda key, *args: keys.append(key)):
        get(d, 'z', optimize_graph=False)
    assert len(keys) == 2
