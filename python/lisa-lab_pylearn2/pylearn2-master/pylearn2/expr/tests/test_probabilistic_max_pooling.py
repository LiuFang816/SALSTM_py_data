from __future__ import print_function

import numpy as np
import warnings

from theano.compat.six.moves import xrange
from theano import config
from theano import function
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams

from pylearn2.expr.probabilistic_max_pooling import max_pool_python
from pylearn2.expr.probabilistic_max_pooling import max_pool_channels_python
from pylearn2.expr.probabilistic_max_pooling import max_pool
from pylearn2.expr.probabilistic_max_pooling import max_pool_channels
from pylearn2.expr.probabilistic_max_pooling import max_pool_b01c
from pylearn2.expr.probabilistic_max_pooling import max_pool_c01b
from pylearn2.expr.probabilistic_max_pooling import max_pool_unstable
from pylearn2.expr.probabilistic_max_pooling import max_pool_softmax_op
from pylearn2.expr.probabilistic_max_pooling import \
    max_pool_softmax_with_bias_op
from pylearn2.testing import no_debug_mode


def check_correctness_channelwise(f):
    """
    Tests that the theano expression emitted by f computes the same values
    as the ground truth python function
    Note: to keep the python version as dead simple as possible (i.e., to make
    sure there are not bugs in the ground truth) it uses the numerically
    unstable verison of softmax. So this test does not work with too big of
    numbers.
    """

    rng = np.random.RandomState([2012, 7, 19])
    batch_size = 5
    pool_size = 4
    n = 3 * pool_size
    zv = rng.randn(batch_size, n).astype(config.floatX) * 1. - 1.5
    top_down_v = rng.randn(batch_size,  n / pool_size).astype(config.floatX)

    p_np, h_np = max_pool_channels_python(zv, pool_size, top_down_v)

    z_th = T.matrix()
    z_th.name = 'z_th'

    top_down_th = T.matrix()
    top_down_th.name = 'top_down_th'

    p_th, h_th = f(z_th, pool_size, top_down_th)

    func = function([z_th, top_down_th], [p_th, h_th])

    pv, hv = func(zv, top_down_v)

    assert p_np.shape == pv.shape
    assert h_np.shape == hv.shape
    if not np.allclose(h_np, hv):
        print((h_np.min(), h_np.max()))
        print((hv.min(), hv.max()))
        assert False
    if not np.allclose(p_np, pv):
        diff = abs(p_np - pv)
        print('max diff ', diff.max())
        print('min diff ', diff.min())
        print('ave diff ', diff.mean())
        assert False


def check_correctness_sigmoid_channelwise(f):
    """
    Tests that f is equivalent to the sigmoid function when the pool size is 1
    """

    rng = np.random.RandomState([2012, 7, 19])
    batch_size = 5
    pool_size = 1
    n = 3 * pool_size
    zv = rng.randn(batch_size, n).astype(config.floatX) * 1. - 1.5
    top_down_v = rng.randn(batch_size,  n / pool_size).astype(config.floatX)

    z_th = T.matrix()
    z_th.name = 'z_th'

    top_down_th = T.matrix()
    top_down_th.name = 'top_down_th'

    p_th, h_th = f(z_th, pool_size, top_down_th)
    h_s = T.nnet.sigmoid(z_th + top_down_th)

    func = function([z_th, top_down_th], [p_th, h_th, h_s])

    pv, hv, h_s = func(zv, top_down_v)
    p_s = h_s

    assert p_s.shape == pv.shape
    assert h_s.shape == hv.shape
    if not np.allclose(h_s, hv):
        print((h_s.min(), h_s.max()))
        print((hv.min(), hv.max()))
        assert False
    if not np.allclose(p_s, pv):
        diff = abs(p_s - pv)
        print('max diff ', diff.max())
        print('min diff ', diff.min())
        print('ave diff ', diff.mean())
        assert False


def check_correctness(f):
    rng = np.random.RandomState([2012, 7, 19])
    batch_size = 5
    rows = 32
    cols = 30
    channels = 3
    pool_rows = 2
    pool_cols = 3
    zv = rng.randn(batch_size, rows, cols,
                   channels).astype(config.floatX) * 2. - 3.

    p_np, h_np = max_pool_python(zv, (pool_rows, pool_cols))

    z_th = T.TensorType(broadcastable=(False, False, False, False),
                        dtype=config.floatX)()
    z_th.name = 'z_th'

    p_th, h_th = f(z_th, (pool_rows, pool_cols))

    func = function([z_th], [p_th, h_th])

    pv, hv = func(zv)

    assert p_np.shape == pv.shape
    assert h_np.shape == hv.shape
    if not np.allclose(h_np, hv):
        print((h_np.min(), h_np.max()))
        print((hv.min(), hv.max()))
        assert False
    assert np.allclose(p_np, pv)


def check_correctness_bc01(f):
    """
    Tests that the theano expression emitted by f computes the same values
    as the ground truth python function
    Note: to keep the python version as dead simple as possible (i.e., to make
    sure there are not bugs in the ground truth) it uses the numerically
    unstable verison of softmax. So this test does not work with too big of
    numbers.
    """

    rng = np.random.RandomState([2012, 7, 19])
    batch_size = 5
    rows = 32
    cols = 30
    channels = 3
    pool_rows = 2
    pool_cols = 3
    zv = rng.randn(batch_size,  rows, cols,
                   channels).astype(config.floatX) * 1. - 1.5
    top_down_v = rng.randn(batch_size, rows / pool_rows, cols / pool_cols,
                           channels).astype(config.floatX)

    p_np, h_np = max_pool_python(zv, (pool_rows, pool_cols), top_down_v)

    z_th = T.TensorType(broadcastable=(False, False, False, False),
                        dtype = config.floatX)()
    z_th.name = 'z_th'
    zr = z_th.dimshuffle(0, 3, 1, 2)

    top_down_th = T.TensorType(broadcastable=(False, False, False, False),
                               dtype = config.floatX)()
    top_down_th.name = 'top_down_th'
    top_down_r = top_down_th.dimshuffle(0, 3, 1, 2)

    p_th, h_th = f(zr, (pool_rows, pool_cols), top_down_r)

    func = function([z_th, top_down_th], [p_th.dimshuffle(0, 2, 3, 1),
                                          h_th.dimshuffle(0, 2, 3, 1)])

    pv, hv = func(zv, top_down_v)

    assert p_np.shape == pv.shape
    assert h_np.shape == hv.shape
    if not np.allclose(h_np, hv):
        print((h_np.min(), h_np.max()))
        print((hv.min(), hv.max()))
        assert False
    if not np.allclose(p_np, pv):
        diff = abs(p_np - pv)
        print('max diff ', diff.max())
        print('min diff ', diff.min())
        print('ave diff ', diff.mean())
        assert False


def check_correctness_c01b(f):
    """
    Tests that the theano expression emitted by f computes the same values
    as the ground truth python function
    Note: to keep the python version as dead simple as possible (i.e., to make
    sure there are not bugs in the ground truth) it uses the numerically
    unstable version of softmax. So this test does not work with too big of
    numbers.
    """

    rng = np.random.RandomState([2013, 5, 6])
    batch_size = 5
    rows = 32
    cols = 30
    channels = 3
    pool_rows = 2
    pool_cols = 3

    # Do the python ground truth in b01c format
    zv = rng.randn(batch_size,  rows, cols,
                   channels).astype(config.floatX) * 1. - 1.5
    top_down_v = rng.randn(batch_size, rows / pool_rows, cols / pool_cols,
                           channels).astype(config.floatX)

    p_np, h_np = max_pool_python(zv, (pool_rows, pool_cols), top_down_v)

    # Dimshuffle the inputs into c01b for the theano implementation
    z_th = T.TensorType(broadcastable=(False, False, False, False),
                        dtype = config.floatX)()
    z_th.tag.test_value = zv
    z_th.name = 'z_th'
    zr = z_th.dimshuffle(3, 1, 2, 0)

    top_down_th = T.TensorType(broadcastable=(False, False, False, False),
                               dtype = config.floatX)()
    top_down_th.name = 'top_down_th'
    top_down_th.tag.test_value = top_down_v
    top_down_r = top_down_th.dimshuffle(3, 1, 2, 0)

    p_th, h_th = f(zr, (pool_rows, pool_cols), top_down_r)

    func = function([z_th, top_down_th], [p_th.dimshuffle(3, 1, 2, 0),
                                          h_th.dimshuffle(3, 1, 2, 0)])

    pv, hv = func(zv, top_down_v)

    if not p_np.shape == pv.shape:
        raise AssertionError(str((p_np.shape, pv.shape)))
    assert h_np.shape == hv.shape
    if not np.allclose(h_np, hv):
        print((h_np.min(), h_np.max()))
        print((hv.min(), hv.max()))
        assert False
    if not np.allclose(p_np, pv):
        diff = abs(p_np - pv)
        print('max diff ', diff.max())
        print('min diff ', diff.min())
        print('ave diff ', diff.mean())
        assert False
    warnings.warn("TODO: make sampling tests run on c01b format of pooling.")


@no_debug_mode
def check_sample_correctishness_b01c(f):
    batch_size = 5
    rows = 32
    cols = 30
    channels = 3
    pool_rows = 2
    pool_cols = 3
    rng = np.random.RandomState([2012, 9, 26])
    zv = rng.randn(batch_size, rows, cols,
                   channels).astype(config.floatX) * 2. - 3.
    top_down_v = rng.randn(batch_size, rows / pool_rows, cols / pool_cols,
                           channels).astype(config.floatX)

    z_th = T.TensorType(broadcastable=(False, False, False, False),
                        dtype = config.floatX)()
    z_th.name = 'z_th'

    top_down_th = T.TensorType(broadcastable=(False, False, False, False),
                               dtype = config.floatX)()
    top_down_th.name = 'top_down_th'

    theano_rng = MRG_RandomStreams(rng.randint(2147462579))
    p_th, h_th, p_sth, h_sth = f(z_th, (pool_rows, pool_cols), top_down_th,
                                 theano_rng)

    prob_func = function([z_th, top_down_th], [p_th, h_th])
    pv, hv = prob_func(zv, top_down_v)

    sample_func = function([z_th, top_down_th], [p_sth, h_sth])

    acc_p = 0. * pv
    acc_h = 0. * hv

    # make sure the test gets good coverage, ie, that it includes many
    # different activation probs for both detector and pooling layer
    buckets = 10
    bucket_width = 1. / float(buckets)
    for i in xrange(buckets):
        lower_lim = i * bucket_width
        upper_lim = (i+1) * bucket_width

        assert np.any((pv >= lower_lim) * (pv < upper_lim))
        assert np.any((hv >= lower_lim) * (hv < upper_lim))

    assert upper_lim == 1.

    for i in xrange(10000):
        ps, hs = sample_func(zv, top_down_v)

        assert ps.shape == pv.shape
        assert hs.shape == hv.shape

        acc_p += ps
        acc_h += hs

    est_p = acc_p / float(i+1)
    est_h = acc_h / float(i+1)

    pd = np.abs(est_p-pv)
    hd = np.abs(est_h-hv)

    """
    # plot maps of the estimation error, this is to see if it has
    # some spatial pattern this is useful for detecting bugs like
    # not handling the border correctly, etc.
    from pylearn2.gui.patch_viewer import PatchViewer

    pv = PatchViewer((pd.shape[0],pd.shape[3]),(pd.shape[1],pd.shape[2]),
                      is_color = False)
    for i in xrange(pd.shape[0]):
    for j in xrange(pd.shape[3]):
    pv.add_patch((pd[i,:,:,j] / pd.max() )* 2.0 - 1.0, rescale = False)
    pv.show()

    pv = PatchViewer((hd.shape[0],hd.shape[3]),(hd.shape[1],hd.shape[2]),
                      is_color = False)
    for i in xrange(hd.shape[0]):
    for j in xrange(hd.shape[3]):
    pv.add_patch( (hd[i,:,:,j] / hd.max() )* 2.0 - 1.0, rescale = False)
    pv.show()
    """

    """
    plot expectation to estimate versus error in estimation
    expect bigger errors for values closer to 0.5

    from matplotlib import pyplot as plt

    #nelem = reduce( lambda x, y : x*y, pd.shape)
    #plt.scatter( pv.reshape(nelem), pd.reshape(nelem))
    #plt.show()

    nelem = reduce( lambda x, y : x*y, hd.shape)
    plt.scatter( hv.reshape(nelem), hd.reshape(nelem))
    plt.show()
    """

    # don't really know how tight this should be
    # but you can try to pose an equivalent problem
    # and implement it in another way
    # using a numpy implementation in softmax_acc.py
    # I got a max error of .17
    assert max(pd.max(), hd.max()) < .17

    # Do exhaustive checks on just the last sample
    assert np.all((ps == 0) + (ps == 1))
    assert np.all((hs == 0) + (hs == 1))

    for k in xrange(batch_size):
        for i in xrange(ps.shape[1]):
            for j in xrange(ps.shape[2]):
                for l in xrange(channels):
                    p = ps[k, i, j, l]
                    h = hs[k, i*pool_rows:(i+1)*pool_rows,
                           j*pool_cols:(j+1)*pool_cols, l]
                    assert h.shape == (pool_rows, pool_cols)
                    assert p == h.max()

    """ If you made it to here, it's correctish
     (cant tell if samples are perfectly "correct") """


@no_debug_mode
def check_sample_correctishness_c01b(f):
    batch_size = 5
    rows = 32
    cols = 30
    channels = 3
    pool_rows = 2
    pool_cols = 3
    rng = np.random.RandomState([2012, 9, 26])
    zv = rng.randn(channels, rows, cols,
                   batch_size).astype(config.floatX) * 2. - 3.
    top_down_v = rng.randn(channels, rows / pool_rows, cols / pool_cols,
                           batch_size).astype(config.floatX)

    z_th = T.TensorType(broadcastable=(False, False, False, False),
                        dtype = config.floatX)()
    z_th.name = 'z_th'
    z_th.tag.test_value = zv

    top_down_th = T.TensorType(broadcastable=(False, False, False, False),
                               dtype = config.floatX)()
    top_down_th.name = 'top_down_th'
    top_down_th.tag.test_value = top_down_v

    theano_rng = MRG_RandomStreams(rng.randint(2147462579))
    p_th, h_th, p_sth, h_sth = f(z_th, (pool_rows, pool_cols), top_down_th,
                                 theano_rng)

    prob_func = function([z_th, top_down_th], [p_th, h_th])
    pv, hv = prob_func(zv, top_down_v)

    sample_func = function([z_th, top_down_th], [p_sth, h_sth])

    acc_p = 0. * pv
    acc_h = 0. * hv

    # make sure the test gets good coverage, ie, that it includes
    # many different activation probs for both detector and pooling layer
    buckets = 10
    bucket_width = 1. / float(buckets)
    for i in xrange(buckets):
        lower_lim = i * bucket_width
        upper_lim = (i+1) * bucket_width

        assert np.any((pv >= lower_lim) * (pv < upper_lim))
        assert np.any((hv >= lower_lim) * (hv < upper_lim))

    assert upper_lim == 1.

    for i in xrange(10000):
        ps, hs = sample_func(zv, top_down_v)

        assert ps.shape == pv.shape
        assert hs.shape == hv.shape

        acc_p += ps
        acc_h += hs

    est_p = acc_p / float(i+1)
    est_h = acc_h / float(i+1)

    pd = np.abs(est_p-pv)
    hd = np.abs(est_h-hv)

    # don't really know how tight this should be
    # but you can try to pose an equivalent problem
    # and implement it in another way
    # using a numpy implementation in softmax_acc.py
    # I got a max error of .17
    assert max(pd.max(), hd.max()) < .17

    # Do exhaustive checks on just the last sample
    assert np.all((ps == 0) + (ps == 1))
    assert np.all((hs == 0) + (hs == 1))

    for k in xrange(batch_size):
        for i in xrange(ps.shape[1]):
            for j in xrange(ps.shape[2]):
                for l in xrange(channels):
                    p = ps[l, i, j, k]
                    h = hs[l, i*pool_rows:(i+1)*pool_rows,
                           j*pool_cols:(j+1)*pool_cols, k]
                    assert h.shape == (pool_rows, pool_cols)
                    assert p == h.max()

    """ If you made it to here, it's correctish
     (cant tell if samples are perfectly "correct") """


@no_debug_mode
def check_sample_correctishness_bc01(f):
    """
    Tests that the sample mean converges to the conditional
    expectation given by the function
    Tests that p really is the max of the samples
    Tests that at most one h in a group is on
    """

    batch_size = 5
    rows = 32
    cols = 30
    channels = 3
    pool_rows = 2
    pool_cols = 3

    rng = np.random.RandomState([2012, 9, 26])
    zv = rng.randn(batch_size, channels, rows,
                   cols).astype(config.floatX) * 2. - 3.
    top_down_v = rng.randn(batch_size, channels, rows / pool_rows,
                           cols / pool_cols).astype(config.floatX)

    z_th = T.TensorType(broadcastable=(False, False, False, False),
                        dtype = config.floatX)()
    z_th.tag.test_value = zv
    z_th.name = 'z_th'

    top_down_th = T.TensorType(broadcastable=(False, False, False, False),
                               dtype = config.floatX)()
    top_down_th.tag.test_value = top_down_v
    top_down_th.name = 'top_down_th'

    theano_rng = MRG_RandomStreams(rng.randint(2147462579))
    p_th, h_th, p_sth, h_sth = f(z_th, (pool_rows, pool_cols), top_down_th,
                                 theano_rng)

    prob_func = function([z_th, top_down_th], [p_th, h_th])
    pv, hv = prob_func(zv, top_down_v)

    sample_func = function([z_th, top_down_th], [p_sth, h_sth])

    acc_p = 0. * pv
    acc_h = 0. * hv

    # make sure the test gets good coverage, ie, that it includes many
    # different activation probs for both detector and pooling layer
    buckets = 10
    bucket_width = 1. / float(buckets)
    for i in xrange(buckets):
        lower_lim = i * bucket_width
        upper_lim = (i+1) * bucket_width

        assert np.any((pv >= lower_lim) * (pv < upper_lim))
        assert np.any((hv >= lower_lim) * (hv < upper_lim))

    assert upper_lim == 1.

    for i in xrange(10000):
        ps, hs = sample_func(zv, top_down_v)

        assert ps.shape == pv.shape
        assert hs.shape == hv.shape

        acc_p += ps
        acc_h += hs

    est_p = acc_p / float(i+1)
    est_h = acc_h / float(i+1)

    pd = np.abs(est_p-pv)
    hd = np.abs(est_h-hv)

    """
    # plot maps of the estimation error, this is to see if it has some
    # spatial pattern this is useful for detecting bugs like not handling
    # the border correctly, etc.
    from pylearn2.gui.patch_viewer import PatchViewer

    pv = PatchViewer((pd.shape[0],pd.shape[3]),(pd.shape[1],pd.shape[2]),
                                                            is_color = False)
    for i in xrange(pd.shape[0]):
    for j in xrange(pd.shape[3]):
    pv.add_patch( (pd[i,:,:,j] / pd.max() )* 2.0 - 1.0, rescale = False)
    pv.show()

    pv = PatchViewer((hd.shape[0],hd.shape[3]), (hd.shape[1],hd.shape[2]),
                                                        is_color = False)
    for i in xrange(hd.shape[0]):
    for j in xrange(hd.shape[3]):
    pv.add_patch( (hd[i,:,:,j] / hd.max() )* 2.0 - 1.0, rescale = False)
    pv.show()
    """

    """
    plot expectation to estimate versus error in estimation
    expect bigger errors for values closer to 0.5

    from matplotlib import pyplot as plt

    #nelem = reduce( lambda x, y : x*y, pd.shape)
    #plt.scatter( pv.reshape(nelem), pd.reshape(nelem))
    #plt.show()

    nelem = reduce( lambda x, y : x*y, hd.shape)
    plt.scatter( hv.reshape(nelem), hd.reshape(nelem))
    plt.show()
    """

    # don't really know how tight this should be
    # but you can try to pose an equivalent problem
    # and implement it in another way
    # using a numpy implementation in softmax_acc.py
    # I got a max error of .17
    assert max(pd.max(), hd.max()) < .17

    # Do exhaustive checks on just the last sample
    assert np.all((ps == 0) + (ps == 1))
    assert np.all((hs == 0) + (hs == 1))

    for k in xrange(batch_size):
        for i in xrange(ps.shape[2]):
            for j in xrange(ps.shape[3]):
                for l in xrange(channels):
                    p = ps[k, l, i, j]
                    h = hs[k, l, i*pool_rows:(i+1)*pool_rows,
                           j*pool_cols:(j+1)*pool_cols]
                    assert h.shape == (pool_rows, pool_cols)
                    assert p == h.max()
                    assert h.sum() <= 1

    """ If you made it to here, it's correctish
     (cant tell if samples are perfectly "correct") """


@no_debug_mode
def check_sample_correctishness_channelwise(f):
    """
    Tests that the sample mean converges to the conditional expectation given
    by the function Tests that p really is the max of the samples tests that
    at most one h in a group is on
    """

    batch_size = 27
    pool_size = 4
    n = pool_size * 21

    rng = np.random.RandomState([2012, 9, 26])
    zv = rng.randn(batch_size, n).astype(config.floatX) * 3.5 - 5.
    top_down_v = rng.randn(batch_size, n / pool_size).astype(config.floatX)

    z_th = T.matrix()
    z_th.tag.test_value = zv
    z_th.name = 'z_th'

    top_down_th = T.matrix()
    top_down_th.tag.test_value = top_down_v
    top_down_th.name = 'top_down_th'

    theano_rng = MRG_RandomStreams(rng.randint(2147462579))
    p_th, h_th, p_sth, h_sth = f(z_th, pool_size, top_down_th, theano_rng)

    prob_func = function([z_th, top_down_th], [p_th, h_th])
    pv, hv = prob_func(zv, top_down_v)

    sample_func = function([z_th, top_down_th], [p_sth, h_sth])

    acc_p = 0. * pv
    acc_h = 0. * hv

    # make sure the test gets good coverage, ie, that it includes
    # many different activation probs for both detector and pooling layer
    buckets = 10
    bucket_width = 1. / float(buckets)
    print(pv.min(), pv.max())
    print(hv.min(), hv.max())
    for i in xrange(buckets):
        lower_lim = i * bucket_width
        upper_lim = (i+1) * bucket_width
        print(lower_lim, upper_lim)

        assert np.any((pv >= lower_lim) * (pv < upper_lim))
        assert np.any((hv >= lower_lim) * (hv < upper_lim))

    assert upper_lim == 1.

    for i in xrange(10000):
        ps, hs = sample_func(zv, top_down_v)

        assert ps.shape == pv.shape
        assert hs.shape == hv.shape

        acc_p += ps
        acc_h += hs

    est_p = acc_p / float(i+1)
    est_h = acc_h / float(i+1)

    pd = np.abs(est_p-pv)
    hd = np.abs(est_h-hv)

    """
    # plot maps of the estimation error, this is to see if it has some
    # spatial pattern this is useful for detecting bugs like not handling
    # the border correctly, etc.
    # from pylearn2.gui.patch_viewer import PatchViewer

    pv = PatchViewer((pd.shape[0],pd.shape[3]),(pd.shape[1],pd.shape[2]),
                                                            is_color = False)
    for i in xrange(pd.shape[0]):
    for j in xrange(pd.shape[3]):
    pv.add_patch( (pd[i,:,:,j] / pd.max() )* 2.0 - 1.0, rescale = False)
    pv.show()

    pv = PatchViewer((hd.shape[0],hd.shape[3]),(hd.shape[1],hd.shape[2]),
                                                            is_color = False)
    for i in xrange(hd.shape[0]):
    for j in xrange(hd.shape[3]):
    pv.add_patch( (hd[i,:,:,j] / hd.max() )* 2.0 - 1.0, rescale = False)
    pv.show()
    """

    """
    plot expectation to estimate versus error in estimation
    expect bigger errors for values closer to 0.5

    from matplotlib import pyplot as plt

    #nelem = reduce( lambda x, y : x*y, pd.shape)
    #plt.scatter( pv.reshape(nelem), pd.reshape(nelem))
    #plt.show()

    nelem = reduce( lambda x, y : x*y, hd.shape)
    plt.scatter( hv.reshape(nelem), hd.reshape(nelem))
    plt.show()
    """

    # don't really know how tight this should be
    # but you can try to pose an equivalent problem
    # and implement it in another way
    # using a numpy implementation in softmax_acc.py
    # I got a max error of .17
    assert max(pd.max(), hd.max()) < .17

    # Do exhaustive checks on just the last sample
    assert np.all((ps == 0) + (ps == 1))
    assert np.all((hs == 0) + (hs == 1))

    for k in xrange(batch_size):
        for i in xrange(ps.shape[1]):
            p = ps[k, i]
            h = hs[k, i*pool_size:(i+1)*pool_size]
            assert h.shape == (pool_size,)
            assert p == h.max()
            assert h.sum() <= 1

    """ If you made it to here, it's correctish
     (cant tell if samples are perfectly "correct") """


def test_max_pool_channels():
    check_correctness_channelwise(max_pool_channels)


def test_max_pool_channels_sigmoid():
    check_correctness_sigmoid_channelwise(max_pool_channels)


def test_max_pool_channels_samples():
    check_sample_correctishness_channelwise(max_pool_channels)


def test_max_pool():
    check_correctness_bc01(max_pool)


def test_max_pool_c01b():
    check_correctness_c01b(max_pool_c01b)


def test_max_pool_samples():
    check_sample_correctishness_bc01(max_pool)


def test_max_pool_b01c_samples():
    check_sample_correctishness_b01c(max_pool_b01c)


def test_max_pool_c01b_samples():
    check_sample_correctishness_c01b(max_pool_c01b)


def test_max_pool_b01c():
    check_correctness(max_pool_b01c)


def test_max_pool_unstable():
    check_correctness(max_pool_unstable)


def test_max_pool_softmax_op():
    check_correctness(max_pool_softmax_op)


def test_max_pool_softmax_with_bias_op():
    check_correctness(max_pool_softmax_with_bias_op)
