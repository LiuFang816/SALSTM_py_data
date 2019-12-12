"""
Helper functions for determining which tests to skip.
"""

__authors__ = "Ian Goodfellow"
__copyright__ = "Copyright 2010-2012, Universite de Montreal"
__credits__ = ["Ian Goodfellow"]
__license__ = "3-clause BSD"
__maintainer__ = "LISA Lab"
__email__ = "pylearn-dev@googlegroups"
from nose.plugins.skip import SkipTest
import os
from theano.sandbox import cuda

scipy_works = True
try:
    import scipy
except ImportError:
    # pyflakes gets mad if you set scipy to None here
    scipy_works = False

sklearn_works = True
try:
    import sklearn
except ImportError:
    sklearn_works = False

h5py_works = True
try:
    import h5py
except ImportError:
    h5py_works = False

matplotlib_works = True
try:
    from matplotlib import pyplot
except ImportError:
    matplotlib_works = False


def skip_if_no_data():
    if 'PYLEARN2_DATA_PATH' not in os.environ:
        raise SkipTest()


def skip_if_no_scipy():
    if not scipy_works:
        raise SkipTest()


def skip_if_no_sklearn():
    if not sklearn_works:
        raise SkipTest()


def skip_if_no_gpu():
    if cuda.cuda_available == False:
        raise SkipTest('Optional package cuda disabled.')


def skip_if_no_h5py():
    if not h5py_works:
        raise SkipTest()


def skip_if_no_matplotlib():
    if not matplotlib_works:
        raise SkipTest("matplotlib and pyplot are not available")
