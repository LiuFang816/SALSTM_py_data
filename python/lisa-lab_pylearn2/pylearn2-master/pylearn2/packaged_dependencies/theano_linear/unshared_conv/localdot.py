"""
WRITEME
"""

import logging
from ..linear import LinearTransform
from .unshared_conv import FilterActs, ImgActs
from theano.compat.six.moves import xrange
from theano.sandbox import cuda
if cuda.cuda_available:
    import gpu_unshared_conv # register optimizations

import numpy as np
import warnings

try:
    import matplotlib.pyplot as plt
except (RuntimeError, ImportError, TypeError) as matplotlib_exception:
    warnings.warn("Unable to import matplotlib. Some features unavailable. "
                  "Original exception: " + str(matplotlib_exception))

logger = logging.getLogger(__name__)


class LocalDot(LinearTransform):
    """
    LocalDot is an linear operation computationally similar to
    convolution in the spatial domain, except that whereas convolution
    applying a single filter or set of filters across an image, the
    LocalDot has different filterbanks for different points in the image.

    Mathematically, this is a general linear transform except for a
    restriction that filters are 0 outside of a spatially localized patch
    within the image.

    Image shape is 5-tuple:
        color_groups
        colors_per_group
        rows
        cols
        images

    Filterbank shape is 7-tuple (!)
        0 row_positions
        1 col_positions
        2 colors_per_group
        3 height
        4 width
        5 color_groups
        6 filters_per_group

    The result of left-multiplication a 5-tuple with shape:
        filter_groups
        filters_per_group
        row_positions
        col_positions
        images

    Parameters
    ----------
    filters : WRITEME
    irows : WRITEME
        Image rows
    icols : WRITEME
        Image columns
    subsample : WRITEME
    padding_start : WRITEME
    filters_shape : WRITEME
    message : WRITEME
    """

    def __init__(self, filters, irows, icols=None,
            subsample=(1, 1),
            padding_start=None,
            filters_shape=None,
            message=""):
        LinearTransform.__init__(self, [filters])
        self._filters = filters
        if filters_shape is None:
            self._filters_shape = tuple(filters.get_value(borrow=True).shape)
        else:
            self._filters_shape = tuple(filters_shape)
        self._irows = irows
        if icols is None:
            self._icols = irows
        else:
            self._icols = icols
        if self._icols != self._irows:
            raise NotImplementedError('GPU code at least needs square imgs')
        self._subsample = tuple(subsample)
        self._padding_start = padding_start

        if len(self._filters_shape) != 7:
            raise TypeError('need 7-tuple filter shape', self._filters_shape)
        if self._subsample[0] != self._subsample[1]:
            raise ValueError('subsampling must be same in rows and cols')

        self._filter_acts = FilterActs(self._subsample[0])
        self._img_acts = ImgActs(module_stride=self._subsample[0])

        if message:
            self._message = message
        else:
            self._message = filters.name

    def rmul(self, x):
        """
        .. todo::

            WRITEME
        """
        assert x.ndim == 5
        return self._filter_acts(x, self._filters)

    def rmul_T(self, x):
        """
        .. todo::

            WRITEME
        """
        return self._img_acts(self._filters, x, self._irows, self._icols)

    def col_shape(self):
        """
        .. todo::

            WRITEME
        """
        ishape = self.row_shape() + (-99,)
        fshape = self._filters_shape
        hshape, = self._filter_acts.infer_shape(None, (ishape, fshape))
        assert hshape[-1] == -99
        return hshape[:-1]

    def row_shape(self):
        """
        .. todo::

            WRITEME
        """
        fshape = self._filters_shape
        fmodulesR, fmodulesC, fcolors, frows, fcols = fshape[:-2]
        fgroups, filters_per_group = fshape[-2:]

        return fgroups, fcolors, self._irows, self._icols


    def print_status(self):
        """
        .. todo::

            WRITEME
        """
        raise NotImplementedError("TODO: fix dependence on non-existent "
                "ndarray_status function")
        """print ndarray_status(
                self._filters.get_value(borrow=True),
                msg='%s{%s}'% (self.__class__.__name__,
                    self._message))
        """

    def imshow_gray(self):
        """
        .. todo::

            WRITEME
        """
        filters = self._filters.get_value()
        modR, modC, colors, rows, cols, grps, fs_per_grp = filters.shape
        logger.info(filters.shape)

        rval = np.zeros((
            modR * (rows + 1) - 1,
            modC * (cols + 1) - 1,
        ))

        for rr, modr in enumerate(xrange(0, rval.shape[0], rows + 1)):
            for cc, modc in enumerate(xrange(0, rval.shape[1], cols + 1)):
                rval[modr:modr + rows, modc:modc + cols] = filters[rr, cc, 0, :, :, 0, 0]

        plt.imshow(rval, cmap='gray')
        return rval
