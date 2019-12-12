"""
Utility functions for working with images.
"""
import logging
import numpy as np
plt = None
axes = None
from theano.compat.six.moves import xrange
from theano.compat.six import string_types
import warnings
try:
    import matplotlib.pyplot as plt
    import matplotlib.axes
except (RuntimeError, ImportError, TypeError) as matplotlib_exception:
    warnings.warn("Unable to import matplotlib. Some features unavailable. "
                  "Original exception: " + str(matplotlib_exception))
import os

try:
    from PIL import Image
except ImportError:
    Image = None

from pylearn2.utils import string_utils as string
from pylearn2.utils.exc import reraise_as
from tempfile import mkstemp
from multiprocessing import Process

import subprocess

logger = logging.getLogger(__name__)


def ensure_Image():
    """Makes sure Image has been imported from PIL"""
    global Image
    if Image is None:
        raise RuntimeError("You are trying to use PIL-dependent functionality"
                           " but don't have PIL installed.")


def imview(*args, **kwargs):
    """
    A matplotlib-based image viewer command,
    wrapping `matplotlib.pyplot.imshow` but behaving more
    sensibly.

    Parameters
    ----------
    figure : TODO
        TODO: write parameters section using decorators to inherit
        the matplotlib docstring

    Notes
    -----
    Parameters are identical to `matplotlib.pyplot.imshow`
    but this behaves somewhat differently:

    * By default, it creates a new figure (unless a
      `figure` keyword argument is supplied.
    * It modifies the axes of that figure to use the
      full frame, without ticks or tick labels.
    * It turns on `nearest` interpolation by default
      (i.e., it does not antialias pixel data). This
      can be overridden with the `interpolation`
      argument as in `imshow`.

    All other arguments and keyword arguments are passed
    on to `imshow`.`
    """
    if 'figure' not in kwargs:
        f = plt.figure()
    else:
        f = kwargs['figure']
    new_ax = matplotlib.axes.Axes(f,
                                  [0, 0, 1, 1],
                                  xticks=[],
                                  yticks=[],
                                  frame_on=False)
    f.delaxes(f.gca())
    f.add_axes(new_ax)
    if len(args) < 5 and 'interpolation' not in kwargs:
        kwargs['interpolation'] = 'nearest'
    plt.imshow(*args, **kwargs)


def imview_async(*args, **kwargs):
    """
    A version of `imview` that forks a separate process and
    immediately shows the image.

    Parameters
    ----------
    window_title : str
        TODO: writeme with decorators to inherit the other imviews'
        docstrings

    Notes
    -----
    Supports the `window_title` keyword argument to cope with
    the title always being 'Figure 1'.

    Returns the `multiprocessing.Process` handle.
    """
    if 'figure' in kwargs:
        raise ValueError("passing a figure argument not supported")

    def fork_image_viewer():
        f = plt.figure()
        kwargs['figure'] = f
        imview(*args, **kwargs)
        if 'window_title' in kwargs:
            f.set_window_title(kwargs['window_title'])
        plt.show()

    p = Process(None, fork_image_viewer)
    p.start()
    return p


def show(image):
    """
    .. todo::

        WRITEME

    Parameters
    ----------
    image : PIL Image object or ndarray
        If ndarray, integer formats are assumed to use 0-255
        and float formats are assumed to use 0-1
    """
    viewer_command = string.preprocess('${PYLEARN2_VIEWER_COMMAND}')

    if viewer_command == 'inline':
        return imview(image)

    if hasattr(image, '__array__'):
        # do some shape checking because PIL just raises a tuple indexing error
        # that doesn't make it very clear what the problem is
        if len(image.shape) < 2 or len(image.shape) > 3:
            raise ValueError('image must have either 2 or 3 dimensions but its'
                             ' shape is ' + str(image.shape))

        # The below is a temporary workaround that prevents us from crashing
        # 3rd party image viewers such as eog by writing out overly large
        # images.
        # In the long run we should determine if this is a bug in PIL when
        # producing
        # such images or a bug in eog and determine a proper fix.
        # Since this is hopefully just a short term workaround the
        # constants below are not included in the interface to the
        # function, so that 3rd party code won't start passing them.
        max_height = 4096
        max_width = 4096

        # Display separate warnings for each direction, since it's
        # common to crop only one.
        if image.shape[0] > max_height:
            image = image[0:max_height, :, :]
            warnings.warn("Cropping image to smaller height to avoid crashing "
                          "the viewer program.")
        if image.shape[0] > max_width:
            image = image[:, 0:max_width, :]
            warnings.warn("Cropping the image to a smaller width to avoid "
                          "crashing the viewer program.")
        # This ends the workaround

        if image.dtype == 'int8':
            image = np.cast['uint8'](image)
        elif str(image.dtype).startswith('float'):
            # don't use *=, we don't want to modify the input array
            image = image * 255.
            image = np.cast['uint8'](image)

        # PIL is too stupid to handle single-channel arrays
        if len(image.shape) == 3 and image.shape[2] == 1:
            image = image[:, :, 0]

        try:
            ensure_Image()
            image = Image.fromarray(image)
        except TypeError:
            reraise_as(TypeError("PIL issued TypeError on ndarray of shape " +
                                 str(image.shape) + " and dtype " +
                                 str(image.dtype)))

    # Create a temporary file with the suffix '.png'.
    fd, name = mkstemp(suffix='.png')
    os.close(fd)

    # Note:
    #   Although we can use tempfile.NamedTemporaryFile() to create
    #   a temporary file, the function should be used with care.
    #
    #   In Python earlier than 2.7, a temporary file created by the
    #   function will be deleted just after the file is closed.
    #   We can re-use the name of the temporary file, but there is an
    #   instant where a file with the name does not exist in the file
    #   system before we re-use the name. This may cause a race
    #   condition.
    #
    #   In Python 2.7 or later, tempfile.NamedTemporaryFile() has
    #   the 'delete' argument which can control whether a temporary
    #   file will be automatically deleted or not. With the argument,
    #   the above race condition can be avoided.
    #

    image.save(name)
    if os.name == 'nt':
        subprocess.Popen(viewer_command + ' ' + name + ' && del ' + name,
                         shell=True)
    else:
        subprocess.Popen(viewer_command + ' ' + name + ' ; rm ' + name,
                         shell=True)


def pil_from_ndarray(ndarray):
    """
    Converts an ndarray to a PIL image.

    Parameters
    ----------
    ndarray : ndarray
        An ndarray containing an image.

    Returns
    -------
    pil : PIL Image
        A PIL Image containing the image.
    """
    try:
        if ndarray.dtype == 'float32' or ndarray.dtype == 'float64':
            assert ndarray.min() >= 0.0
            assert ndarray.max() <= 1.0

            ndarray = np.cast['uint8'](ndarray * 255)

            if len(ndarray.shape) == 3 and ndarray.shape[2] == 1:
                ndarray = ndarray[:, :, 0]

        ensure_Image()
        rval = Image.fromarray(ndarray)
        return rval
    except Exception as e:
        logger.exception('original exception: ')
        logger.exception(e)
        logger.exception('ndarray.dtype: {0}'.format(ndarray.dtype))
        logger.exception('ndarray.shape: {0}'.format(ndarray.shape))
        raise

    assert False


def ndarray_from_pil(pil, dtype='uint8'):
    """
    Converts a PIL Image to an ndarray.

    Parameters
    ----------
    pil : PIL Image
        An image represented as a PIL Image object
    dtype : str
        The dtype of ndarray to create

    Returns
    -------
    ndarray : ndarray
        The image as an ndarray.
    """
    rval = np.asarray(pil)

    if dtype != rval.dtype:
        rval = np.cast[dtype](rval)

    if str(dtype).startswith('float'):
        rval /= 255.

    if len(rval.shape) == 2:
        rval = rval.reshape(rval.shape[0], rval.shape[1], 1)

    return rval


def rescale(image, shape):
    """
    Scales image to be no larger than shape. PIL might give you
    unexpected results beyond that.

    Parameters
    ----------
    image : WRITEME
    shape : WRITEME

    Returns
    -------
    WRITEME
    """

    assert len(image.shape) == 3  # rows, cols, channels
    assert len(shape) == 2  # rows, cols

    i = pil_from_ndarray(image)

    ensure_Image()
    i.thumbnail([shape[1], shape[0]], Image.ANTIALIAS)

    rval = ndarray_from_pil(i, dtype=image.dtype)

    return rval
resize = rescale


def fit_inside(image, shape):
    """
    Scales image down to fit inside shape preserves proportions of image

    Parameters
    ----------
    image : WRITEME
    shape : WRITEME

    Returns
    -------
    WRITEME
    """

    assert len(image.shape) == 3  # rows, cols, channels
    assert len(shape) == 2  # rows, cols

    if image.shape[0] <= shape[0] and image.shape[1] <= shape[1]:
        return image.copy()

    row_ratio = float(image.shape[0]) / float(shape[0])
    col_ratio = float(image.shape[1]) / float(shape[1])

    if row_ratio > col_ratio:
        target_shape = [shape[0], min(image.shape[1] / row_ratio, shape[1])]
    else:
        target_shape = [min(image.shape[0] / col_ratio, shape[0]), shape[1]]

    assert target_shape[0] <= shape[0]
    assert target_shape[1] <= shape[1]
    assert target_shape[0] == shape[0] or target_shape[1] == shape[1]
    rval = rescale(image, target_shape)
    return rval


def letterbox(image, shape):
    """
    Pads image with black letterboxing to bring image.shape up to shape

    Parameters
    ----------
    image : WRITEME
    shape : WRITEME

    Returns
    -------
    WRITEME
    """

    assert len(image.shape) == 3  # rows, cols, channels
    assert len(shape) == 2  # rows, cols

    assert image.shape[0] <= shape[0]
    assert image.shape[1] <= shape[1]

    if image.shape[0] == shape[0] and image.shape[1] == shape[1]:
        return image.copy()

    rval = np.zeros((shape[0], shape[1], image.shape[2]), dtype=image.dtype)

    rstart = (shape[0] - image.shape[0]) / 2
    cstart = (shape[1] - image.shape[1]) / 2

    rend = rstart + image.shape[0]
    cend = cstart + image.shape[1]
    rval[rstart:rend, cstart:cend] = image

    return rval


def make_letterboxed_thumbnail(image, shape):
    """
    Scales image down to shape. Preserves proportions of image, introduces
    black letterboxing if necessary.

    Parameters
    ----------
    image : WRITEME
    shape : WRITEME

    Returns
    -------
    WRITEME
    """

    assert len(image.shape) == 3
    assert len(shape) == 2

    shrunk = fit_inside(image, shape)
    letterboxed = letterbox(shrunk, shape)

    return letterboxed


def load(filepath, rescale_image=True, dtype='float64'):
    """
    Load an image from a file.

    Parameters
    ----------
    filepath : str
        Path to the image file to load
    rescale_image : bool
        Default value: True
        If True, returned images have pixel values in [0, 1]. Otherwise,
        values are in [0, 255].
    dtype: str
        The dtype to use for the returned value

    Returns
    -------
    img : numpy ndarray
        An array containing the image that was in the file.
    """
    assert isinstance(filepath, string_types)

    if not rescale_image and dtype == 'uint8':
        ensure_Image()
        rval = np.asarray(Image.open(filepath))
        assert rval.dtype == 'uint8'
        return rval

    s = 1.0
    if rescale_image:
        s = 255.
    try:
        ensure_Image()
        rval = Image.open(filepath)
    except Exception:
        reraise_as(Exception("Could not open " + filepath))

    numpy_rval = np.array(rval)

    msg = ("Tried to load an image, got an array with %d"
           " dimensions. Expected 2 or 3."
           "This may indicate a mildly corrupted image file. Try "
           "converting it to a different image format with a different "
           "editor like gimp or imagemagic. Sometimes these programs are "
           "more robust to minor corruption than PIL and will emit a "
           "correctly formatted image in the new format.")
    if numpy_rval.ndim not in [2, 3]:
        logger.error(dir(rval))
        logger.error(rval)
        logger.error(rval.size)
        rval.show()
        raise AssertionError(msg % numpy_rval.ndim)
    rval = numpy_rval

    rval = np.cast[dtype](rval) / s

    if rval.ndim == 2:
        rval = rval.reshape(rval.shape[0], rval.shape[1], 1)

    if rval.ndim != 3:
        raise AssertionError("Something went wrong opening " +
                             filepath + '. Resulting shape is ' +
                             str(rval.shape) +
                             " (it's meant to have 3 dimensions by now)")

    return rval


def save(filepath, ndarray):
    """
    Saves an image to a file.

    Parameters
    ----------
    filepath : str
        The path to write the file to.
    ndarray : ndarray
        An array containing the image to be saved.
    """
    pil_from_ndarray(ndarray).save(filepath)


def scale_to_unit_interval(ndar, eps=1e-8):
    """
    Scales all values in the ndarray ndar to be between 0 and 1

    Parameters
    ----------
    ndar : WRITEME
    eps : WRITEME

    Returns
    -------
    WRITEME
    """
    ndar = ndar.copy()
    ndar -= ndar.min()
    ndar *= 1.0 / (ndar.max() + eps)
    return ndar


def tile_raster_images(X, img_shape, tile_shape, tile_spacing=(0, 0),
                       scale_rows_to_unit_interval=True,
                       output_pixel_vals=True):
    """
    Transform an array with one flattened image per row, into an array in
    which images are reshaped and layed out like tiles on a floor.

    This function is useful for visualizing datasets whose rows are images,
    and also columns of matrices for transforming those rows
    (such as the first layer of a neural net).

    Parameters
    ----------
    x : numpy.ndarray
        2-d ndarray or 4 tuple of 2-d ndarrays or None for channels,
        in which every row is a flattened image.

    shape : 2-tuple of ints
        The first component is the height of each image,
        the second component is the width.

    tile_shape : 2-tuple of ints
        The number of images to tile in (row, columns) form.

    scale_rows_to_unit_interval : bool
        Whether or not the values need to be before being plotted to [0, 1].

    output_pixel_vals : bool
        Whether or not the output should be pixel values (int8) or floats.

    Returns
    -------
    y : 2d-ndarray
        The return value has the same dtype as X, and is suitable for
        viewing as an image with PIL.Image.fromarray.
    """

    assert len(img_shape) == 2
    assert len(tile_shape) == 2
    assert len(tile_spacing) == 2

    # The expression below can be re-written in a more C style as
    # follows :
    #
    # out_shape    = [0,0]
    # out_shape[0] = (img_shape[0]+tile_spacing[0])*tile_shape[0] -
    #                tile_spacing[0]
    # out_shape[1] = (img_shape[1]+tile_spacing[1])*tile_shape[1] -
    #                tile_spacing[1]
    out_shape = [(ishp + tsp) * tshp - tsp for ishp, tshp, tsp
                 in zip(img_shape, tile_shape, tile_spacing)]

    if isinstance(X, tuple):
        assert len(X) == 4
        # Create an output np ndarray to store the image
        if output_pixel_vals:
            out_array = np.zeros((out_shape[0], out_shape[1], 4),
                                 dtype='uint8')
        else:
            out_array = np.zeros((out_shape[0], out_shape[1], 4),
                                 dtype=X.dtype)

        # colors default to 0, alpha defaults to 1 (opaque)
        if output_pixel_vals:
            channel_defaults = [0, 0, 0, 255]
        else:
            channel_defaults = [0., 0., 0., 1.]

        for i in xrange(4):
            if X[i] is None:
                # if channel is None, fill it with zeros of the correct
                # dtype
                dt = out_array.dtype
                if output_pixel_vals:
                    dt = 'uint8'
                out_array[:, :, i] = np.zeros(out_shape, dtype=dt) + \
                    channel_defaults[i]
            else:
                # use a recurrent call to compute the channel and store it
                # in the output
                out_array[:, :, i] = tile_raster_images(
                    X[i], img_shape, tile_shape, tile_spacing,
                    scale_rows_to_unit_interval, output_pixel_vals)
        return out_array

    else:
        # if we are dealing with only one channel
        H, W = img_shape
        Hs, Ws = tile_spacing

        # generate a matrix to store the output
        dt = X.dtype
        if output_pixel_vals:
            dt = 'uint8'
        out_array = np.zeros(out_shape, dtype=dt)

        for tile_row in xrange(tile_shape[0]):
            for tile_col in xrange(tile_shape[1]):
                if tile_row * tile_shape[1] + tile_col < X.shape[0]:
                    this_x = X[tile_row * tile_shape[1] + tile_col]
                    if scale_rows_to_unit_interval:
                        # if we should scale values to be between 0 and 1
                        # do this by calling the `scale_to_unit_interval`
                        # function
                        this_img = scale_to_unit_interval(
                            this_x.reshape(img_shape))
                    else:
                        this_img = this_x.reshape(img_shape)
                    # add the slice to the corresponding position in the
                    # output array
                    c = 1
                    if output_pixel_vals:
                        c = 255
                    out_array[
                        tile_row * (H + Hs): tile_row * (H + Hs) + H,
                        tile_col * (W + Ws): tile_col * (W + Ws) + W
                        ] = this_img * c
        return out_array


if __name__ == '__main__':
    black = np.zeros((50, 50, 3), dtype='uint8')

    red = black.copy()
    red[:, :, 0] = 255

    green = black.copy()
    green[:, :, 1] = 255

    show(black)
    show(green)
    show(red)
