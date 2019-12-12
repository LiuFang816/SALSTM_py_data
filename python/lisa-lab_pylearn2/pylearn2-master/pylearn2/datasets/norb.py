"""
An interface to the small NORB dataset. Unlike `./norb_small.py`, this reads
the original NORB file format, not the LISA lab's `.npy` version.

Currently only supports the Small NORB Dataset.

Download the dataset from
`here <http://www.cs.nyu.edu/~ylclab/data/norb-v1.0-small/>`_.

NORB dataset(s) by Fu Jie Huang and Yann LeCun.
"""

__authors__ = "Guillaume Desjardins and Matthew Koichi Grimes"
__copyright__ = "Copyright 2010-2014, Universite de Montreal"
__credits__ = __authors__.split(" and ")
__license__ = "3-clause BSD"
__maintainer__ = "Matthew Koichi Grimes"
__email__ = "mkg alum mit edu (@..)"


import bz2
import gzip
import logging
import os
import warnings
try:
    from exceptions import DeprecationWarning
except ImportError:
    pass

import numpy
import theano


from pylearn2.datasets import dense_design_matrix
from pylearn2.datasets.cache import datasetCache
from pylearn2.space import VectorSpace, Conv2DSpace, CompositeSpace

from pylearn2.datasets.new_norb import StereoViewConverter

logger = logging.getLogger(__name__)

warnings.warn("Using deprecated module pylearn2.datasets.norb. "
              "This will be replaced with pylearn2.datasets.new_norb in "
              "December 2014. Users are encouraged to switch to that "
              "module now.", DeprecationWarning)


class SmallNORB(dense_design_matrix.DenseDesignMatrix):

    """
    An interface to the small NORB dataset.

    If instantiated with default arguments, target labels are integers
    representing categories, which can be looked up using

      category_name = SmallNORB.get_category(label).

    If instantiated with multi_target=True, labels are vectors of indices
    representing:

      [ category, instance, elevation, azimuth, lighting ]

    Like with category, there are class methods that map these ints to their
    actual values, e.g:

      category = SmallNORB.get_category(label[0])
      elevation = SmallNORB.get_elevation_degrees(label[2])

    Parameters
    ----------
    which_set: str
        Must be 'train' or 'test'.
    multi_target: bool, optional
        If False, each label is an integer labeling the image catergory. If
        True, each label is a vector: [category, instance, lighting, elevation,
        azimuth]. All labels are given as integers. Use the categories,
        elevation_degrees, and azimuth_degrees arrays to map from these
        integers to actual values.
    """

    # Actual image shape may change, e.g. after being preprocessed by
    # datasets.preprocessing.Downsample
    original_image_shape = (96, 96)

    _categories = ['animal',  # four-legged animal
                   'human',  # human figure
                   'airplane',
                   'truck',
                   'car']

    @classmethod
    def get_category(cls, scalar_label):
        """
        Returns the category string corresponding to an integer category label.
        """
        return cls._categories[int(scalar_label)]

    @classmethod
    def get_elevation_degrees(cls, scalar_label):
        """
        Returns the elevation, in degrees, corresponding to an integer
        elevation label.
        """
        scalar_label = int(scalar_label)
        assert scalar_label >= 0
        assert scalar_label < 9
        return 30 + 5 * scalar_label

    @classmethod
    def get_azimuth_degrees(cls, scalar_label):
        """
        Returns the azimuth, in degrees, corresponding to an integer
        label.
        """
        scalar_label = int(scalar_label)
        assert scalar_label >= 0
        assert scalar_label <= 34
        assert (scalar_label % 2) == 0
        return scalar_label * 10

    # Maps azimuth labels (ints) to their actual values, in degrees.
    azimuth_degrees = numpy.arange(0, 341, 20)

    # Maps a label type to its index within a label vector.
    label_type_to_index = {'category': 0,
                           'instance': 1,
                           'elevation': 2,
                           'azimuth': 3,
                           'lighting': 4}

    # Number of labels, for each label type.
    num_labels_by_type = (len(_categories),
                          10,  # instances
                          9,   # elevations
                          18,  # azimuths
                          6)   # lighting

    # [mkg] Dropped support for the 'center' argument for now. In Pylearn 1, it
    # shifted the pixel values from [0:255] by subtracting 127.5. Seems like a
    # form of preprocessing, which might be better implemented separately using
    # the Preprocess class.
    def __init__(self, which_set, multi_target=False, stop=None):
        """
        parameters
        ----------

        which_set : str
            Must be 'train' or 'test'.

        multi_target : bool
            If False, each label is an integer labeling the image catergory. If
            True, each label is a vector: [category, instance, lighting,
            elevation, azimuth]. All labels are given as integers. Use the
            categories, elevation_degrees, and azimuth_degrees arrays to map
            from these integers to actual values.
        """

        assert which_set in ['train', 'test']

        self.which_set = which_set

        subtensor = slice(0, stop) if stop is not None else None

        X = SmallNORB.load(which_set, 'dat', subtensor=subtensor)

        # Casts to the GPU-supported float type, using theano._asarray(), a
        # safer alternative to numpy.asarray().
        #
        # TODO: move the dtype-casting to the view_converter's output space,
        #       once dtypes-for-spaces is merged into master.
        X = theano._asarray(X, theano.config.floatX)

        # Formats data as rows in a matrix, for DenseDesignMatrix
        X = X.reshape(-1, 2 * numpy.prod(self.original_image_shape))

        # This is uint8
        y = SmallNORB.load(which_set, 'cat', subtensor=subtensor)
        if multi_target:
            y_extra = SmallNORB.load(which_set, 'info', subtensor=subtensor)
            y = numpy.hstack((y[:, numpy.newaxis], y_extra))

        datum_shape = ((2, ) +  # two stereo images
                       self.original_image_shape +
                       (1, ))  # one color channel

        # 's' is the stereo channel: 0 (left) or 1 (right)
        axes = ('b', 's', 0, 1, 'c')
        view_converter = StereoViewConverter(datum_shape, axes)

        super(SmallNORB, self).__init__(X=X,
                                        y=y,
                                        y_labels=numpy.max(y) + 1,
                                        view_converter=view_converter)

    @classmethod
    def load(cls, which_set, filetype, subtensor):
        """Reads and returns a single file as a numpy array."""

        assert which_set in ['train', 'test']
        assert filetype in ['dat', 'cat', 'info']

        def getPath(which_set):
            dirname = os.path.join(os.getenv('PYLEARN2_DATA_PATH'),
                                   'norb_small/original')
            if which_set == 'train':
                instance_list = '46789'
            elif which_set == 'test':
                instance_list = '01235'

            filename = 'smallnorb-5x%sx9x18x6x2x96x96-%s-%s.mat' % \
                (instance_list, which_set + 'ing', filetype)

            return os.path.join(dirname, filename)

        def parseNORBFile(file_handle, subtensor=None, debug=False):
            """
            Load all or part of file 'file_handle' into a numpy ndarray

            .. todo::

                WRITEME properly

            :param file_handle: file from which to read file can be opended
              with open(), gzip.open() and bz2.BZ2File()
              @type file_handle: file-like object. Can be a gzip open file.

            :param subtensor: If subtensor is not None, it should be like the
              argument to numpy.ndarray.__getitem__.  The following two
              expressions should return equivalent ndarray objects, but the one
              on the left may be faster and more memory efficient if the
              underlying file f is big.

              read(file_handle, subtensor) <===> read(file_handle)[*subtensor]

              Support for subtensors is currently spotty, so check the code to
              see if your particular type of subtensor is supported.
              """

            def readNums(file_handle, num_type, count):
                """
                Reads 4 bytes from file, returns it as a 32-bit integer.
                """
                num_bytes = count * numpy.dtype(num_type).itemsize
                string = file_handle.read(num_bytes)
                return numpy.fromstring(string, dtype=num_type)

            def readHeader(file_handle, debug=False, from_gzip=None):
                """
                .. todo::

                    WRITEME properly

                :param file_handle: an open file handle.
                :type file_handle: a file or gzip.GzipFile object

                :param from_gzip: bool or None
                :type from_gzip: if None determine the type of file handle.

                :returns: data type, element size, rank, shape, size
                """

                if from_gzip is None:
                    from_gzip = isinstance(file_handle,
                                           (gzip.GzipFile, bz2.BZ2File))

                key_to_type = {0x1E3D4C51: ('float32', 4),
                               # what is a packed matrix?
                               # 0x1E3D4C52: ('packed matrix', 0),
                               0x1E3D4C53: ('float64', 8),
                               0x1E3D4C54: ('int32', 4),
                               0x1E3D4C55: ('uint8', 1),
                               0x1E3D4C56: ('int16', 2)}

                type_key = readNums(file_handle, 'int32', 1)[0]
                elem_type, elem_size = key_to_type[type_key]
                if debug:
                    logger.debug("header's type key, type, type size: "
                                 "{0} {1} {2}".format(type_key, elem_type,
                                                      elem_size))
                if elem_type == 'packed matrix':
                    raise NotImplementedError('packed matrix not supported')

                num_dims = readNums(file_handle, 'int32', 1)[0]
                if debug:
                    logger.debug('# of dimensions, according to header: '
                                 '{0}'.format(num_dims))

                if from_gzip:
                    shape = readNums(file_handle,
                                     'int32',
                                     max(num_dims, 3))[:num_dims]
                else:
                    shape = numpy.fromfile(file_handle,
                                           dtype='int32',
                                           count=max(num_dims, 3))[:num_dims]

                if debug:
                    logger.debug('Tensor shape, as listed in header: '
                                 '{0}'.format(shape))

                return elem_type, elem_size, shape

            elem_type, elem_size, shape = readHeader(file_handle, debug)
            beginning = file_handle.tell()

            num_elems = numpy.prod(shape)

            result = None
            if isinstance(file_handle, (gzip.GzipFile, bz2.BZ2File)):
                assert subtensor is None, \
                    "Subtensors on gzip files are not implemented."
                result = readNums(file_handle,
                                  elem_type,
                                  num_elems * elem_size).reshape(shape)
            elif subtensor is None:
                result = numpy.fromfile(file_handle,
                                        dtype=elem_type,
                                        count=num_elems).reshape(shape)
            elif isinstance(subtensor, slice):
                if subtensor.step not in (None, 1):
                    raise NotImplementedError('slice with step',
                                              subtensor.step)
                if subtensor.start not in (None, 0):
                    bytes_per_row = numpy.prod(shape[1:]) * elem_size
                    file_handle.seek(
                        beginning + subtensor.start * bytes_per_row)
                shape[0] = min(shape[0], subtensor.stop) - subtensor.start
                num_elems = numpy.prod(shape)
                result = numpy.fromfile(file_handle,
                                        dtype=elem_type,
                                        count=num_elems).reshape(shape)
            else:
                raise NotImplementedError('subtensor access not written yet:',
                                          subtensor)

            return result
        fname = getPath(which_set)
        fname = datasetCache.cache_file(fname)
        file_handle = open(fname)

        return parseNORBFile(file_handle, subtensor)

    def get_topological_view(self, mat=None, single_tensor=True):
        """
        .. todo::

            WRITEME
        """
        result = super(SmallNORB, self).get_topological_view(mat)

        if single_tensor:
            warnings.warn("The single_tensor argument is True by default to "
                          "maintain backwards compatibility. This argument "
                          "will be removed, and the behavior will become that "
                          "of single_tensor=False, as of August 2014.")
            axes = list(self.view_converter.axes)
            s_index = axes.index('s')
            assert axes.index('b') == 0
            num_image_pairs = result[0].shape[0]
            shape = (num_image_pairs, ) + self.view_converter.shape

            # inserts a singleton dimension where the 's' dimesion will be
            mono_shape = shape[:s_index] + (1, ) + shape[(s_index + 1):]

            for i, res in enumerate(result):
                logger.info("result {0} shape: {1}".format(i, str(res.shape)))

            result = tuple(t.reshape(mono_shape) for t in result)
            result = numpy.concatenate(result, axis=s_index)
        else:
            warnings.warn("The single_tensor argument will be removed on "
                          "August 2014. The behavior will be the same as "
                          "single_tensor=False.")

        return result
