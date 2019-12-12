"""
An implementation of probabilistic max-pooling, based on

"Convolutional Deep Belief Networks for Scalable
Unsupervised Learning of Hierarchical Representations"
Honglak Lee, Roger Grosse, Rajesh Ranganath, and Andrew Y. Ng
ICML 2009


This paper defines probabilistic max-pooling in the context
of a Convolutional Deep Belief Network (its energy function is
more like a DBM than a DBN but it is trained like a DBN). Here
we define probabilistic max pooling as a general layer for
use in an energy-based model regardless of how the rest of the
model is assembled.
"""

__authors__ = "Ian Goodfellow"
__copyright__ = "Copyright 2010-2012, Universite de Montreal"
__credits__ = ["Ian Goodfellow"]
__license__ = "3-clause BSD"
__maintainer__ = "LISA Lab"
__email__ = "pylearn-dev@googlegroups"

import logging
import theano.tensor as T
import numpy as np
from theano.compat.six.moves import xrange
from theano import config
from theano import function
import time
from pylearn2.utils import sharedX
from theano.sandbox.rng_mrg import MRG_RandomStreams
from theano.gof.op import get_debug_values

logger = logging.getLogger(__name__)


def max_pool(z, pool_shape, top_down=None, theano_rng=None):
    """
    Probabilistic max-pooling

    Parameters
    ----------
    z : theano 4-tensor
        a theano 4-tensor representing input from below
    pool_shape : tuple
        tuple of ints. the shape of regions to be pooled
    top_down : theano 4-tensor, optional
        a theano 4-tensor representing input from above
        if None, assumes top-down input is 0
    theano_rng : MRG_RandomStreams, optional
        Used for random numbers for sampling

    Returns
    -------
    p : theano 4-tensor
        the expected value of the pooling layer p
    h : theano 4-tensor
        the expected value of the detector layer h
    p_samples : theano 4-tensor, only returned if theano_rng is not None
        samples of the pooling layer
    h_samples : theano 4-tensor, only returned if theano_rng is not None
        samples of the detector layer

    Notes
    ------

    all 4-tensors are formatted with axes ('b', 'c', 0, 1).
    This is for maximum speed when using theano's conv2d
    to generate z and top_down, or when using it to infer conditionals of
    other layers using the return values.

    Detailed description:

    Suppose you have a variable h that lives in a Conv2DSpace h_space and
    you want to pool it down to a variable p that lives in a smaller
    Conv2DSpace p.

    This function does that, using non-overlapping pools.

    Specifically, consider one channel of h. h must have a height that is a
    multiple of pool_shape[0] and a width that is a multiple of pool_shape[1].
    A channel of h can thus be broken down into non-overlapping rectangles
    of shape pool_shape.

    Now consider one rectangular pooled region within one channel of h.
    I now use 'h' to refer just to this rectangle, and 'p' to refer to
    just the one pooling unit associated with that rectangle.
    We assume that the space that h and p live in is constrained such
    that h and p are both binary and p = max(h). To reduce the state-space
    in order to make probabilistic computations cheaper we also
    constrain sum(h) <= 1.
    Suppose h contains k different units. Suppose that the only term
    in the model's energy function involving h is -(z*h).sum()
    (elemwise multiplication) and the only term in
    the model's energy function involving p is -(top_down*p).sum().

    Then P(h[i] = 1) = softmax( [ z[1], z[2], ..., z[k], -top_down] )[i]
    and P(p = 1) = 1-softmax( [z[1], z[2], ..., z[k], -top_down])[k]

    This variation of the function assumes that z, top_down, and all
    return values use Conv2D axes ('b', 'c', 0, 1).
    This variation of the function implements the softmax using a
    theano graph of exp, maximum, sub, and div operations.

    Performance notes:
    It might be possible to make a faster implementation with different
    theano ops. rather than using set_subtensor, it might be possible
    to use the stuff in theano.sandbox.neighbours. Probably not possible,
    or at least nasty, because that code isn't written with multiple
    channels in mind, and I don't think just a reshape can fix it.
    Some work on this in galatea.cond.neighbs.py
    At some point images2neighbs' gradient was broken so check that
    it has been fixed before sinking too much time into this.

    Stabilizing the softmax is also another source of slowness.
    Here it is stabilized with several calls to maximum and sub.
    It might also be possible to stabilize it with
    T.maximum(-top_down,T.signal.downsample.max_pool(z)).
    Don't know if that would be faster or slower.

    Elsewhere in this file I implemented the softmax with a reshape
    and call to Softmax / SoftmaxWithBias.
    This is slower, even though Softmax is faster on the GPU than the
    equivalent max/sub/exp/div graph. Maybe the reshape is too expensive.

    Benchmarks show that most of the time is spent in GpuIncSubtensor
    when running on gpu. So it is mostly that which needs a faster
    implementation. One other way to implement this would be with
    a linear.Conv2D.lmul_T, where the convolution stride is equal to
    the pool width, and the thing to multiply with is the hparts stacked
    along the channel axis. Unfortunately, conv2D doesn't work right
    with stride > 2 and is pretty slow for stride 2. Conv3D is used to
    mitigate some of this, but only has CPU code.
    """

    z_name = z.name
    if z_name is None:
        z_name = 'anon_z'

    batch_size, ch, zr, zc = z.shape

    r, c = pool_shape

    zpart = []

    mx = None

    if top_down is None:
        t = 0.
    else:
        t = - top_down
        t.name = 'neg_top_down'

    for i in xrange(r):
        zpart.append([])
        for j in xrange(c):
            cur_part = z[:, :, i:zr:r, j:zc:c]
            if z_name is not None:
                cur_part.name = z_name + '[%d,%d]' % (i, j)
            zpart[i].append(cur_part)
            if mx is None:
                mx = T.maximum(t, cur_part)
                if cur_part.name is not None:
                    mx.name = 'max(-top_down,' + cur_part.name + ')'
            else:
                max_name = None
                if cur_part.name is not None:
                    mx_name = 'max(' + cur_part.name + ',' + mx.name + ')'
                mx = T.maximum(mx, cur_part)
                mx.name = mx_name
    mx.name = 'local_max(' + z_name + ')'

    pt = []

    for i in xrange(r):
        pt.append([])
        for j in xrange(c):
            z_ij = zpart[i][j]
            safe = z_ij - mx
            safe.name = 'safe_z(%s)' % z_ij.name
            cur_pt = T.exp(safe)
            cur_pt.name = 'pt(%s)' % z_ij.name
            pt[-1].append(cur_pt)

    off_pt = T.exp(t - mx)
    off_pt.name = 'p_tilde_off(%s)' % z_name
    denom = off_pt

    for i in xrange(r):
        for j in xrange(c):
            denom = denom + pt[i][j]
    denom.name = 'denom(%s)' % z_name

    off_prob = off_pt / denom
    p = 1. - off_prob
    p.name = 'p(%s)' % z_name

    hpart = []
    for i in xrange(r):
        hpart.append([pt_ij / denom for pt_ij in pt[i]])

    h = T.alloc(0., batch_size, ch, zr, zc)

    for i in xrange(r):
        for j in xrange(c):
            h.name = 'h_interm'
            h = T.set_subtensor(h[:, :, i:zr:r, j:zc:c], hpart[i][j])

    h.name = 'h(%s)' % z_name

    if theano_rng is None:
        return p, h
    else:
        events = []
        for i in xrange(r):
            for j in xrange(c):
                events.append(hpart[i][j])
        events.append(off_prob)

        events = [event.dimshuffle(0, 1, 2, 3, 'x') for event in events]

        events = tuple(events)

        stacked_events = T.concatenate(events, axis=4)

        rows = zr // pool_shape[0]
        cols = zc // pool_shape[1]
        outcomes = pool_shape[0] * pool_shape[1] + 1
        assert stacked_events.ndim == 5
        for se, bs, r, c, chv in get_debug_values(stacked_events, batch_size,
                                                  rows, cols, ch):
            assert se.shape[0] == bs
            assert se.shape[1] == r
            assert se.shape[2] == c
            assert se.shape[3] == chv
            assert se.shape[4] == outcomes
        reshaped_events = stacked_events.reshape((
            batch_size * rows * cols * ch, outcomes))

        multinomial = theano_rng.multinomial(pvals=reshaped_events,
                                             dtype=p.dtype)

        reshaped_multinomial = multinomial.reshape((batch_size, ch, rows,
                                                    cols, outcomes))

        h_sample = T.alloc(0., batch_size, ch, zr, zc)

        idx = 0
        for i in xrange(r):
            for j in xrange(c):
                h_sample = T.set_subtensor(h_sample[:, :, i:zr:r, j:zc:c],
                                           reshaped_multinomial[:, :, :, :,
                                           idx])
                idx += 1

        p_sample = 1 - reshaped_multinomial[:, :, :, :, -1]

        return p, h, p_sample, h_sample


def max_pool_c01b(z, pool_shape, top_down=None, theano_rng=None):
    """
    .. todo::

        WRITEME properly

    Like max_pool but with all 4-tensors formatted with axes ('c', 0, 1, 'b').
    This is for maximum speed when using-cuda convnet.

    Notes
    -----
    Performance notes:
    Stabilizing the softmax is one source of slowness. Here it is stabilized
    with several calls to maximum and sub. It might also be possible to
    stabilize it with T.maximum(-top_down,<cuda convnet max pooling>).
    Don't know if that would be faster or slower.

    Benchmarks show that most of the time is spent in GpuIncSubtensor
    when running on gpu. So it is mostly that which needs a faster
    implementation. One other way to implement this would be with cuda
    convnet convolution, where the convolution stride is equal to the
    pool width, and the thing to multiply with is the hparts stacked
    along the channel axis. This isn't a feasible solution for max_pool
    because of theano convolution's poor support for strides, but for cuda
    convnet it could give a speedup.
    """

    z_name = z.name
    if z_name is None:
        z_name = 'anon_z'

    ch, zr, zc, batch_size = z.shape

    r, c = pool_shape

    zpart = []

    mx = None

    if top_down is None:
        t = 0.
    else:
        t = - top_down
        t.name = 'neg_top_down'

    for i in xrange(r):
        zpart.append([])
        for j in xrange(c):
            cur_part = z[:, i:zr:r, j:zc:c, :]
            if z_name is not None:
                cur_part.name = z_name + '[%d, %d]' % (i, j)
            zpart[i].append(cur_part)
            if mx is None:
                mx = T.maximum(t, cur_part)
                if cur_part.name is not None:
                    mx.name = 'max(-top_down,' + cur_part.name + ')'
            else:
                max_name = None
                if cur_part.name is not None:
                    mx_name = 'max(' + cur_part.name + ',' + mx.name + ')'
                mx = T.maximum(mx, cur_part)
                mx.name = mx_name
    mx.name = 'local_max(' + z_name + ')'

    pt = []

    for i in xrange(r):
        pt.append([])
        for j in xrange(c):
            z_ij = zpart[i][j]
            safe = z_ij - mx
            safe.name = 'safe_z(%s)' % z_ij.name
            cur_pt = T.exp(safe)
            cur_pt.name = 'pt(%s)' % z_ij.name
            pt[-1].append(cur_pt)

    off_pt = T.exp(t - mx)
    off_pt.name = 'p_tilde_off(%s)' % z_name
    denom = off_pt

    for i in xrange(r):
        for j in xrange(c):
            denom = denom + pt[i][j]
    denom.name = 'denom(%s)' % z_name

    off_prob = off_pt / denom
    p = 1. - off_prob
    p.name = 'p(%s)' % z_name

    hpart = []
    for i in xrange(r):
        hpart.append([pt_ij / denom for pt_ij in pt[i]])

    h = T.alloc(0., ch, zr, zc, batch_size)

    for i in xrange(r):
        for j in xrange(c):
            h.name = 'h_interm'
            h = T.set_subtensor(h[:, i:zr:r, j:zc:c, :], hpart[i][j])

    h.name = 'h(%s)' % z_name

    if theano_rng is None:
        return p, h
    else:
        events = []
        for i in xrange(r):
            for j in xrange(c):
                events.append(hpart[i][j])
        events.append(off_prob)

        events = [event.dimshuffle(0, 1, 2, 3, 'x') for event in events]

        events = tuple(events)

        stacked_events = T.concatenate(events, axis=4)

        ch, rows, cols, batch_size, outcomes = stacked_events.shape
        reshaped_events = stacked_events.reshape((ch * rows * cols *
                                                  batch_size, outcomes))

        multinomial = theano_rng.multinomial(pvals=reshaped_events,
                                             dtype=p.dtype)

        reshaped_multinomial = multinomial.reshape((ch, rows, cols, batch_size,
                                                    outcomes))

        h_sample = T.alloc(0., ch, zr, zc, batch_size)

        idx = 0
        for i in xrange(r):
            for j in xrange(c):
                h_sample = T.set_subtensor(h_sample[:, i:zr:r, j:zc:c, :],
                                           reshaped_multinomial[:, :, :, :,
                                           idx])
                idx += 1

        p_sample = 1 - reshaped_multinomial[:, :, :, :, -1]

        return p, h, p_sample, h_sample


def max_pool_channels(z, pool_size, top_down=None, theano_rng=None):
    """
    Unlike Honglak's convolutional max pooling, which pools over spatial
    locations within each channels, this does max pooling in a densely
    connected model. Here we pool groups of channels together.

    Parameters
    ----------
    z : theano matrix
        representings a batch of input from below
    pool_size : int
        the number of features to combine into one pooled unit
    top_down : theano matrix, optional
        a theano matrix representing input from above
        if None, assumes top-down input is 0
    theano_rng : MRG_RandomStreams, optional
        For random numbers for sampling

    Returns
    -------
    h : theano matrix
        a theano matrix for the expected value of the detector layer h
    p : theano matrix
        a theano matrix for the expected value of the pooling layer p
    h_samples : theano matrix, only returned if theano_rng is not None
        a theano matrix of samples of the detector layer
    p_samples: theano matrix, only returned if theano_rng is not None
        a theano matrix of samples of the pooling layer

    Notes
    -----
    All matrices are formatted as (num_example, num_features)
    """

    z_name = z.name
    if z_name is None:
        z_name = 'anon_z'

    if pool_size == 1:
        if top_down is None:
            top_down = 0.
        total_input = z + top_down
        p = T.nnet.sigmoid(total_input)
        h = p

        if theano_rng is None:
            return p, h
        else:
            p_samples = theano_rng.binomial(p=p, size=p.shape,
                                            dtype=p.dtype, n=1)
            h_samples = p_samples
            return p_samples, h_samples, p_samples, h_samples
    else:
        batch_size, n = z.shape

        mx = None

        if top_down is None:
            t = 0.
        else:
            t = - top_down
            t.name = 'neg_top_down'

        zpart = []
        for i in xrange(pool_size):
            cur_part = z[:, i:n:pool_size]
            if z_name is not None:
                cur_part.name = z_name + '[%d]' % (i)
            zpart.append(cur_part)
            if mx is None:
                mx = T.maximum(t, cur_part)
                if cur_part.name is not None:
                    mx.name = 'max(-top_down,' + cur_part.name + ')'
            else:
                max_name = None
                if cur_part.name is not None:
                    mx_name = 'max(' + cur_part.name + ',' + mx.name + ')'
                mx = T.maximum(mx, cur_part)
                mx.name = mx_name
        mx.name = 'local_max(' + z_name + ')'

        pt = []

        for i in xrange(pool_size):
            z_i = zpart[i]
            safe = z_i - mx
            safe.name = 'safe_z(%s)' % z_i.name
            cur_pt = T.exp(safe)
            cur_pt.name = 'pt(%s)' % z_i.name
            assert cur_pt.ndim == 2
            pt.append(cur_pt)

        off_pt = T.exp(t - mx)
        assert off_pt.ndim == 2
        off_pt.name = 'p_tilde_off(%s)' % z_name

        denom = off_pt
        for i in xrange(pool_size):
            denom = denom + pt[i]
        assert denom.ndim == 2
        denom.name = 'denom(%s)' % z_name

        off_prob = off_pt / denom
        p = 1. - off_prob
        assert p.dtype == z.dtype

        hpart = [pt_i / denom for pt_i in pt]

        h = T.alloc(0., batch_size, n)

        for i in xrange(pool_size):
            h.name = 'h_interm'
            hp = hpart[i]
            sub_h = h[:, i:n:pool_size]
            assert sub_h.ndim == 2
            assert hp.ndim == 2
            for hv, hsv, hpartv in get_debug_values(h, sub_h, hp):
                logger.info(hv.shape)
                logger.info(hsv.shape)
                logger.info(hpartv.shape)
            h = T.set_subtensor(sub_h, hp)

    p.name = 'p(%s)' % z_name
    h.name = 'h(%s)' % z_name

    if theano_rng is None:
        return p, h
    else:
        events = []
        for i in xrange(pool_size):
            events.append(hpart[i])
        events.append(off_prob)

        events = [event.dimshuffle(0, 1, 'x') for event in events]

        events = tuple(events)

        stacked_events = T.concatenate(events, axis=2)

        outcomes = pool_size + 1
        reshaped_events = stacked_events.reshape((batch_size * n // pool_size,
                                                  outcomes))

        multinomial = theano_rng.multinomial(pvals=reshaped_events,
                                             dtype=p.dtype)

        reshaped_multinomial = multinomial.reshape((batch_size,
                                                    n // pool_size,
                                                    outcomes))

        h_sample = T.zeros_like(z)

        idx = 0
        for i in xrange(pool_size):
            h_sample = T.set_subtensor(h_sample[:, i:n:pool_size],
                                       reshaped_multinomial[:, :, idx])
            idx += 1

        p_sample = 1 - reshaped_multinomial[:, :, -1]

        assert h_sample.dtype == z.dtype

        return p, h, p_sample, h_sample


def max_pool_python(z, pool_shape, top_down=None):
    """
    .. todo::

        WRITEME properly

    Slow python implementation of max_pool
    for unit tests.
    Also, this uses the ('b', 0, 1, 'c') format.
    """

    batch_size, zr, zc, ch = z.shape

    r, c = pool_shape

    assert zr % r == 0
    assert zc % c == 0

    h = np.zeros(z.shape, dtype=z.dtype)
    p = np.zeros((batch_size, zr / r, zc / c, ch), dtype=z.dtype)
    if top_down is None:
        top_down = p.copy()

    for u in xrange(0, zr, r):
        for l in xrange(0, zc, c):
            pt = np.exp(z[:, u:u+r, l:l+c, :])
            off_pt = np.exp(-top_down[:, u/r, l/c, :])
            denom = pt.sum(axis=1).sum(axis=1) + off_pt
            p[:, u/r, l/c, :] = 1. - off_pt / denom
            for i in xrange(batch_size):
                for j in xrange(ch):
                    pt[i, :, :, j] /= denom[i, j]
            h[:, u:u+r, l:l+c, :] = pt

    return p, h


def max_pool_channels_python(z, pool_size, top_down=None):
    """
    .. todo::

        WRITEME properly

    Slow python implementation of max_pool_channels
    for unit tests.
    Also, this uses the ('b', 0, 1, 'c') format.
    """

    batch_size, n = z.shape

    assert n % pool_size == 0

    h = np.zeros(z.shape, dtype=z.dtype)
    p = np.zeros((batch_size, n / pool_size), dtype=z.dtype)
    if top_down is None:
        top_down = p.copy()

    for i in xrange(0, n // pool_size):
        pt = np.exp(z[:, i*pool_size:(i+1)*pool_size])
        off_pt = np.exp(-top_down[:, i])
        denom = pt.sum(axis=1) + off_pt
        assert denom.ndim == 1
        p[:, i] = 1. - off_pt / denom
        for j in xrange(batch_size):
            for k in xrange(pool_size):
                h[j, i*pool_size+k] = pt[j, k] / denom[j]

    return p, h


def max_pool_unstable(z, pool_shape):
    """
    .. todo::

        WRITEME properly

    A version of max_pool that does not numerically stabilize the softmax.
    This is faster, but prone to both overflow and underflow in the
    intermediate computations.
    Mostly useful for benchmarking, to determine how much speedup we
    could hope to get by using a better stabilization method.
    Also, this uses the ('b', 0, 1, 'c') format.
    """

    batch_size, zr, zc, ch = z.shape

    r, c = pool_shape

    zpart = []

    for i in xrange(r):
        zpart.append([])
        for j in xrange(c):
            zpart[i].append(z[:, i:zr:r, j:zc:c, :])

    pt = []

    for i in xrange(r):
        pt.append([T.exp(z_ij) for z_ij in zpart[i]])

    denom = 1.

    for i in xrange(r):
        for j in xrange(c):
            denom = denom + pt[i][j]

    p = 1. - 1. / denom

    hpart = []
    for i in xrange(r):
        hpart.append([pt_ij / denom for pt_ij in pt[i]])

    h = T.alloc(0., batch_size, zr, zc, ch)

    for i in xrange(r):
        for j in xrange(c):
            h = T.set_subtensor(h[:, i:zr:r, j:zc:c, :], hpart[i][j])

    return p, h


def max_pool_b01c(z, pool_shape, top_down=None, theano_rng=None):
    """
    .. todo::

        WRITEME properly

    An implementation of max_pool but where all 4-tensors use the
    ('b', 0, 1, 'c') format.
    """

    z_name = z.name
    if z_name is None:
        z_name = 'anon_z'

    batch_size, zr, zc, ch = z.shape

    r, c = pool_shape

    zpart = []

    mx = None

    if top_down is None:
        t = 0.
    else:
        t = - top_down

    for i in xrange(r):
        zpart.append([])
        for j in xrange(c):
            cur_part = z[:, i:zr:r, j:zc:c, :]
            if z_name is not None:
                cur_part.name = z_name + '[%d, %d]' % (i, j)
            zpart[i].append(cur_part)
            if mx is None:
                mx = T.maximum(t, cur_part)
                if cur_part.name is not None:
                    mx.name = 'max(-top_down,' + cur_part.name + ')'
            else:
                max_name = None
                if cur_part.name is not None:
                    mx_name = 'max(' + cur_part.name + ','+mx.name + ')'
                mx = T.maximum(mx, cur_part)
                mx.name = mx_name
    mx.name = 'local_max('+z_name+')'

    pt = []

    for i in xrange(r):
        pt.append([])
        for j in xrange(c):
            z_ij = zpart[i][j]
            safe = z_ij - mx
            safe.name = 'safe_z(%s)' % z_ij.name
            cur_pt = T.exp(safe)
            cur_pt.name = 'pt(%s)' % z_ij.name
            pt[-1].append(cur_pt)

    off_pt = T.exp(t - mx)
    off_pt.name = 'p_tilde_off(%s)' % z_name
    denom = off_pt

    for i in xrange(r):
        for j in xrange(c):
            denom = denom + pt[i][j]
    denom.name = 'denom(%s)' % z_name

    off_prob = off_pt / denom
    p = 1. - off_prob
    p.name = 'p(%s)' % z_name

    hpart = []
    for i in xrange(r):
        hpart.append([pt_ij / denom for pt_ij in pt[i]])

    h = T.alloc(0., batch_size, zr, zc, ch)

    for i in xrange(r):
        for j in xrange(c):
            h = T.set_subtensor(h[:, i:zr:r, j:zc:c, :], hpart[i][j])

    h.name = 'h(%s)' % z_name

    if theano_rng is None:
        return p, h
    else:
        events = []
        for i in xrange(r):
            for j in xrange(c):
                events.append(hpart[i][j])
        events.append(off_prob)

        events = [event.dimshuffle(0, 1, 2, 3, 'x') for event in events]

        events = tuple(events)

        stacked_events = T.concatenate(events, axis=4)

        batch_size, rows, cols, channels, outcomes = stacked_events.shape
        reshaped_events = stacked_events.reshape((batch_size * rows * cols *
                                                  channels, outcomes))

        multinomial = theano_rng.multinomial(pvals=reshaped_events,
                                             dtype=p.dtype)

        reshaped_multinomial = multinomial.reshape((batch_size, rows, cols,
                                                    channels, outcomes))

        h_sample = T.alloc(0., batch_size, zr, zc, ch)

        idx = 0
        for i in xrange(r):
            for j in xrange(c):
                h_sample = T.set_subtensor(h_sample[:, i:zr:r, j:zc:c, :],
                                           reshaped_multinomial[:, :, :, :,
                                           idx])
                idx += 1

        p_sample = 1 - reshaped_multinomial[:, :, :, :, -1]

        return p, h, p_sample, h_sample


def max_pool_softmax_with_bias_op(z, pool_shape):
    """
    .. todo::

        WRITEME properly

    An implementation of max_pool that uses the SoftmaxWithBias op.
    Mostly kept around for comparison benchmarking purposes.
    Also, this uses the ('b', 0, 1, 'c') format.
    """

    z_name = z.name
    if z_name is None:
        z_name = 'anon_z'

    batch_size, zr, zc, ch = z.shape

    r, c = pool_shape

    flat_z = []

    for i in xrange(r):
        for j in xrange(c):
            cur_part = z[:, i:zr:r, j:zc:c, :]
            assert cur_part.ndim == 4
            if z_name is not None:
                cur_part.name = z_name + '[%d,%d]' % (i, j)
            flat_z.append(cur_part.dimshuffle(0, 1, 2, 3, 'x'))

    flat_z.append(T.zeros_like(flat_z[-1]))

    stacked_z = T.concatenate(flat_z, axis=4)

    batch_size, rows, cols, channels, outcomes = stacked_z.shape
    reshaped_z = stacked_z.reshape((batch_size * rows * cols * channels,
                                    outcomes))

    dist = T.nnet.softmax_with_bias(reshaped_z, T.zeros_like(reshaped_z[0, :]))

    dist = dist.reshape((batch_size, rows, cols, channels, outcomes))

    p = 1. - dist[:, :, :, :, -1]
    p.name = 'p(%s)' % z_name

    h = T.alloc(0., batch_size, zr, zc, ch)

    idx = 0
    for i in xrange(r):
        for j in xrange(c):
            h = T.set_subtensor(h[:, i:zr:r, j:zc:c, :],
                                dist[:, :, :, :, idx])
            idx += 1

    h.name = 'h(%s)' % z_name

    return p, h


def max_pool_softmax_op(z, pool_shape):
    """
    .. todo::

        WRITEME properly

    An implementation of max_pool that uses the SoftmaxWithBias op.
    Mostly kept around for comparison benchmarking purposes.
    Also, this uses the ('b', 0, 1, 'c') format.
    """

    z_name = z.name
    if z_name is None:
        z_name = 'anon_z'

    batch_size, zr, zc, ch = z.shape

    r, c = pool_shape

    flat_z = []

    for i in xrange(r):
        for j in xrange(c):
            cur_part = z[:, i:zr:r, j:zc:c, :]
            assert cur_part.ndim == 4
            if z_name is not None:
                cur_part.name = z_name + '[%d,%d]' % (i, j)
            flat_z.append(cur_part.dimshuffle(0, 1, 2, 3, 'x'))

    flat_z.append(T.zeros_like(flat_z[-1]))

    stacked_z = T.concatenate(flat_z, axis=4)

    batch_size, rows, cols, channels, outcomes = stacked_z.shape
    reshaped_z = stacked_z.reshape((batch_size * rows * cols * channels,
                                    outcomes))

    dist = T.nnet.softmax(reshaped_z)

    dist = dist.reshape((batch_size, rows, cols, channels, outcomes))

    p = 1. - dist[:, :, :, :, len(flat_z)-1]
    p.name = 'p(%s)' % z_name

    h = T.alloc(0., batch_size, zr, zc, ch)

    idx = 0
    for i in xrange(r):
        for j in xrange(c):
            h = T.set_subtensor(h[:, i:zr:r, j:zc:c, :],
                                dist[:, :, :, :, idx])
            idx += 1

    h.name = 'h(%s)' % z_name

    return p, h


def profile(f):
    """
    .. todo::

        WRITEME
    """
    logger.info('profiling {0}'.format(f))
    rng = np.random.RandomState([2012, 7, 19])
    batch_size = 80
    rows = 26
    cols = 27
    channels = 30
    pool_rows = 2
    pool_cols = 3
    zv = rng.randn(batch_size, rows, cols, channels).astype(config.floatX)

    #put the inputs + outputs in shared variables so we don't pay GPU
    # transfer during test
    p_shared = sharedX(zv[:, 0:rows:pool_rows, 0:cols:pool_cols, :])
    h_shared = sharedX(zv)
    z_shared = sharedX(zv)

    p_th, h_th = f(z_shared, (pool_rows, pool_cols))

    func = function([], updates={p_shared: p_th, h_shared: h_th})

    logger.info('warming up')
    for i in xrange(10):
        func()

    trials = 10
    results = []

    for i in xrange(trials):
        t1 = time.time()
        for j in xrange(10):
            func()
        t2 = time.time()
        logger.info(t2 - t1)
        results.append(t2-t1)
    logger.info('final: {0}'.format(sum(results)/float(trials)))


def profile_bc01(f):
    """
    .. todo::

        WRITEME
    """
    logger.info('profiling {0}'.format(f))
    rng = np.random.RandomState([2012, 7, 19])
    batch_size = 80
    rows = 26
    cols = 27
    channels = 30
    pool_rows = 2
    pool_cols = 3
    zv = rng.randn(batch_size, channels, rows, cols).astype(config.floatX)

    # put the inputs + outputs in shared variables so we don't pay GPU
    # transfer during test
    p_shared = sharedX(zv[:, :, 0:rows:pool_rows, 0:cols:pool_cols])
    h_shared = sharedX(zv)
    z_shared = sharedX(zv)

    p_th, h_th = f(z_shared, (pool_rows, pool_cols))

    func = function([], updates={p_shared: p_th, h_shared: h_th})

    logger.info('warming up')
    for i in xrange(10):
        func()

    trials = 10
    results = []

    for i in xrange(trials):
        t1 = time.time()
        for j in xrange(10):
            func()
        t2 = time.time()
        logger.info(t2 - t1)
        results.append(t2-t1)
    logger.info('final: {0}'.format(sum(results)/float(trials)))


def profile_samples(f):
    """
    .. todo::

        WRITEME
    """
    logger.info('profiling samples {0}'.format(f))
    rng = np.random.RandomState([2012, 7, 19])
    theano_rng = MRG_RandomStreams(rng.randint(2147462579))
    batch_size = 80
    rows = 26
    cols = 27
    channels = 30
    pool_rows = 2
    pool_cols = 3
    zv = rng.randn(batch_size, rows, cols, channels).astype(config.floatX)

    #put the inputs + outputs in shared variables so we don't pay GPU
    # transfer during test
    p_shared = sharedX(zv[:, 0:rows:pool_rows, 0:cols:pool_cols, :])
    h_shared = sharedX(zv)
    z_shared = sharedX(zv)

    p_th, h_th, ps_th, hs_th = f(z_shared, (pool_rows, pool_cols), theano_rng)

    func = function([], updates={p_shared: ps_th, h_shared: hs_th})

    logger.info('warming up')
    for i in xrange(10):
        func()

    trials = 10
    results = []

    for i in xrange(trials):
        t1 = time.time()
        for j in xrange(10):
            func()
        t2 = time.time()
        logger.info(t2 - t1)
        results.append(t2-t1)
    logger.info('final: {0}'.format(sum(results)/float(trials)))


def profile_grad(f):
    """
    .. todo::

        WRITEME
    """
    logger.info('profiling gradient of {0}'.format(f))
    rng = np.random.RandomState([2012, 7, 19])
    batch_size = 80
    rows = 26
    cols = 27
    channels = 30
    pool_rows = 2
    pool_cols = 3
    zv = rng.randn(batch_size, rows, cols, channels).astype(config.floatX)

    #put the inputs + outputs in shared variables so we don't pay GPU
    # transfer during test
    grad_shared = sharedX(zv)
    z_shared = sharedX(zv)

    p_th, h_th = f(z_shared, (pool_rows, pool_cols))

    func = function([], updates={grad_shared: T.grad(p_th.sum() + h_th.sum(),
                                 z_shared)})

    logger.info('warming up')
    for i in xrange(10):
        func()

    trials = 10
    results = []

    for i in xrange(trials):
        t1 = time.time()
        for j in xrange(10):
            func()
        t2 = time.time()
        logger.info(t2 - t1)
        results.append(t2-t1)
    logger.info('final: {0}'.format(sum(results)/float(trials)))


def profile_grad_bc01(f):
    """
    .. todo::

        WRITEME
    """
    logger.info('profiling gradient of {0}'.format(f))
    rng = np.random.RandomState([2012, 7, 19])
    batch_size = 80
    rows = 26
    cols = 27
    channels = 30
    pool_rows = 2
    pool_cols = 3
    zv = rng.randn(batch_size, channels, rows, cols).astype(config.floatX)

    # put the inputs + outputs in shared variables so we don't pay GPU
    # transfer during test
    grad_shared = sharedX(zv)
    z_shared = sharedX(zv)

    p_th, h_th = f(z_shared, (pool_rows, pool_cols))

    func = function([], updates={grad_shared: T.grad(p_th.sum() + h_th.sum(),
                                 z_shared)})

    logger.info('warming up')
    for i in xrange(10):
        func()

    trials = 10
    results = []

    for i in xrange(trials):
        t1 = time.time()
        for j in xrange(10):
            func()
        t2 = time.time()
        logger.info(t2 - t1)
        results.append(t2-t1)
    logger.info('final: {0}'.format(sum(results)/float(trials)))


if __name__ == '__main__':
    profile_bc01(max_pool)
    profile_grad_bc01(max_pool)
    """
    profile(max_pool_unstable)
    profile_samples(max_pool_b01c)
    profile(max_pool_softmax_op)
    profile(max_pool_softmax_with_bias_op)
    profile_grad(max_pool_unstable)
    profile_grad(max_pool_b01c)
    profile_grad(max_pool_softmax_op)
    profile_grad(max_pool_softmax_with_bias_op)
    """
