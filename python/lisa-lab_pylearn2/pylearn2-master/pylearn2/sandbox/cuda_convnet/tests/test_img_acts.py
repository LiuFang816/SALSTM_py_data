from __future__ import print_function

__authors__ = "David Warde-Farley, Ian Goodfellow"
__copyright__ = "Copyright 2010-2012, Universite de Montreal"
__credits__ = ["David Warde-Farley", "Ian Goodfellow"]
__license__ = "3-clause BSD"
__maintainer__ = "David Warde-Farley"
__email__ = "wardefar@iro"

from pylearn2.testing.skip import skip_if_no_gpu
skip_if_no_gpu()

import numpy as np
import warnings

from theano import function
from theano.sandbox.cuda import gpu_from_host
from theano.sandbox.cuda import host_from_gpu
from theano.sandbox.rng_mrg import MRG_RandomStreams
from theano import shared
from theano import tensor as T
from theano.tensor import as_tensor_variable
from theano.tensor.nnet.conv import conv2d

from pylearn2.sandbox.cuda_convnet.img_acts import ImageActs

def test_match_full_conv():

    # Tests that running ImageActs with no padding is the same as running
    # theano's conv2D in full mode after flipping the kernel and tranposing
    # the output and input channels
    # In other words, if convolution computes H=XK, we now compute
    # R=HK^T

    rng = np.random.RandomState([2013, 1, 29])

    batch_size = 2
    rows = 6
    cols = 7
    channels = 3
    filter_rows = 5
    filter_cols = filter_rows
    num_filters = 16

    hid_acts = shared(rng.uniform(-1., 1., (num_filters,
                                            rows - filter_rows + 1,
                                            cols - filter_cols + 1,
                                            batch_size)
    ).astype('float32'), name='hidacts')

    filters = shared(rng.uniform(-1., 1., (channels, filter_rows,
        filter_cols, num_filters)).astype('float32'), name='filters')

    gpu_images = gpu_from_host(hid_acts)
    gpu_filters = gpu_from_host(filters)

    output = ImageActs()(gpu_images, gpu_filters, as_tensor_variable((6, 7)))
    output = host_from_gpu(output)

    images_bc01 = hid_acts.dimshuffle(3,0,1,2)
    filters_bc01 = filters.dimshuffle(3,0,1,2)
    # need to tranpose the kernel stack to do imgActs rather than filterActs
    filters_bc01 = filters_bc01.dimshuffle(1, 0, 2, 3)
    # In order to do the transpose operation, we must flip the kernels
    # But in theano's conv2d, the kernels get flipped anyway
    # so in this case, we do not flip the kernel

    output_conv2d = conv2d(images_bc01, filters_bc01, border_mode='full')

    output_conv2d = output_conv2d.dimshuffle(1,2,3,0)

    f = function([], [output, output_conv2d])

    output, output_conv2d = f()

    warnings.warn("""test_match_full_conv success criterion is not very strict. Can we verify that this is OK?
                     One possibility is that theano is numerically unstable and Alex's code is better.
                     Probably theano CPU 64 bit is OK but it's worth checking the others.""")
    if np.abs(output - output_conv2d).max() > 2.4e-6:
        assert type(output) == type(output_conv2d)
        assert output.dtype == output_conv2d.dtype
        if output.shape != output_conv2d.shape:
            print('cuda-convnet shape: ',output.shape)
            print('theano shape: ',output_conv2d.shape)
            assert False
        err = np.abs(output - output_conv2d)
        print('absolute error range: ', (err.min(), err.max()))
        print('mean absolute error: ', err.mean())
        print('cuda-convnet value range: ', (output.min(), output.max()))
        print('theano value range: ', (output_conv2d.min(), output_conv2d.max()))
        assert False

def test_match_full_conv_grad():

    # Tests that the gradient of ImageActs with no padding is the same as the
    # gradient of
    # theano's conv2D in full mode after flipping the kernel and tranposing
    # the output and input channels

    rng = np.random.RandomState([2013, 1, 29])

    batch_size = 2
    rows = 6
    cols = 7
    channels = 3
    filter_rows = 5
    filter_cols = filter_rows
    num_filters = 16

    hid_acts = shared(rng.uniform(-1., 1., (num_filters,
                                            rows - filter_rows + 1,
                                            cols - filter_cols + 1,
                                            batch_size)
    ).astype('float32'), name='hidacts')

    filters = shared(rng.uniform(-1., 1., (channels, filter_rows,
        filter_cols, num_filters)).astype('float32'), name='filters')

    gpu_images = gpu_from_host(hid_acts)
    gpu_filters = gpu_from_host(filters)

    output = ImageActs()(gpu_images, gpu_filters, as_tensor_variable((6, 7)))
    output = host_from_gpu(output)

    images_bc01 = hid_acts.dimshuffle(3,0,1,2)
    filters_bc01 = filters.dimshuffle(3,0,1,2)
    # need to tranpose the kernel stack to do imgActs rather than filterActs
    filters_bc01 = filters_bc01.dimshuffle(1, 0, 2, 3)
    # In order to do the transpose operation, we must flip the kernels
    # But in theano's conv2d, the kernels get flipped anyway
    # so in this case, we do not flip the kernel

    output_conv2d = conv2d(images_bc01, filters_bc01, border_mode='full')

    output_conv2d = output_conv2d.dimshuffle(1,2,3,0)

    theano_rng = MRG_RandomStreams(5 * 10 * 2013)

    random = theano_rng.normal(size=output_conv2d.shape, dtype=output_conv2d.dtype)

    projected = (output * random).sum()
    projected_conv_2d = (output_conv2d * random).sum()

    grads = T.grad(projected, [hid_acts, filters]) + T.grad(projected_conv_2d, [hid_acts, filters])

    f = function([], grads)

    gi, gf, gi_th, gf_th = f()

    assert gi.shape == gi_th.shape
    diff = np.abs(gi - gi_th).max()
    if diff > 2.9e-6:
        assert False

    diff = np.abs(gf - gf_th).max()
    if diff > 1.5e-6:
        raise AssertionError(diff)




if __name__ == '__main__':
    test_match_full_conv()

