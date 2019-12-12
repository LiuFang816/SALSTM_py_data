"""Code for reformatting supervised learning targets."""
from operator import mul

import numpy as np
from theano.compat.six.moves import reduce
import theano.sparse
if theano.sparse.enable_sparse:
    scipy_available = True
    import scipy.sparse
else:
    scipy_available = False
from theano import tensor, config
from pylearn2.utils.exc import reraise_as


class OneHotFormatter(object):
    """
    A target formatter that transforms labels from integers in both single
    and batch mode.

    Parameters
    ----------
    max_labels : int
        The number of possible classes/labels. This means that all labels
        should be < max_labels. Example: For MNIST there are 10 numbers
        and hence max_labels = 10.
    dtype : dtype, optional
        The desired dtype for the converted one-hot vectors. Defaults to
        `config.floatX` if not given.
    """
    def __init__(self, max_labels, dtype=None):
        """
        Initializes the formatter given the number of max labels.
        """
        try:
            np.empty(max_labels)
        except (ValueError, TypeError):
            reraise_as(ValueError("%s got bad max_labels argument '%s'" %
                                  (self.__class__.__name__, str(max_labels))))
        self._max_labels = max_labels
        if dtype is None:
            self._dtype = config.floatX
        else:
            try:
                np.dtype(dtype)
            except TypeError:
                reraise_as(TypeError("%s got bad dtype identifier %s" %
                                     (self.__class__.__name__, str(dtype))))
            self._dtype = dtype

    def format(self, targets, mode='stack', sparse=False):
        """
        Formats a given array of target labels into a one-hot
        vector. If labels appear multiple times, their value
        in the one-hot vector is incremented.

        Parameters
        ----------
        targets : ndarray
            A 1D array of targets, or a batch (2D array) where
            each row is a list of targets.
        mode : string
            The way in which to convert the labels to arrays. Takes
            three different options:

              - "concatenate" : concatenates the one-hot vectors from
                multiple labels
              - "stack" : returns a matrix where each row is the
                one-hot vector of a label
              - "merge" : merges the one-hot vectors together to
                form a vector where the elements are
                the result of an indicator function
                NB: As the result of an indicator function
                the result is the same in case a label
                is duplicated in the input.
        sparse : bool
            If true then the return value is sparse matrix. Note that
            if sparse is True, then mode cannot be 'stack' because
            sparse matrices need to be 2D

        Returns
        -------
        one_hot : a NumPy array (can be 1D-3D depending on settings)
            where normally the first axis are the different batch items,
            the second axis the labels, the third axis the one_hot
            vectors. Can be dense or sparse.
        """
        if mode not in ('concatenate', 'stack', 'merge'):
            raise ValueError("%s got bad mode argument '%s'" %
                             (self.__class__.__name__, str(self._max_labels)))
        elif mode == 'stack' and sparse:
            raise ValueError("Sparse matrices need to be 2D, hence they"
                             "cannot be stacked")
        if targets.ndim > 2:
            raise ValueError("Targets needs to be 1D or 2D, but received %d "
                             "dimensions" % targets.ndim)
        if 'int' not in str(targets.dtype):
            raise TypeError("need an integer array for targets")
        if sparse:
            if not scipy_available:
                raise RuntimeError("The converting of indices to a sparse "
                                   "one-hot vector requires scipy to be "
                                   "installed")
            if mode == 'concatenate':
                one_hot = scipy.sparse.csr_matrix(
                    (np.ones(targets.size, dtype=self._dtype),
                     (targets.flatten() + np.arange(targets.size)
                      * self._max_labels)
                     % (self._max_labels * targets.shape[1]),
                     np.arange(targets.shape[0] + 1) * targets.shape[1]),
                    (targets.shape[0], self._max_labels * targets.shape[1])
                )
            elif mode == 'merge':
                one_hot = scipy.sparse.csr_matrix(
                    (np.ones(targets.size), targets.flatten(),
                     np.arange(targets.shape[0] + 1) * targets.shape[1]),
                    (targets.shape[0], self._max_labels)
                )
        else:
            one_hot = np.zeros(targets.shape + (self._max_labels,),
                               dtype=self._dtype)
            shape = (np.prod(one_hot.shape[:-1]), one_hot.shape[-1])
            one_hot.reshape(shape)[np.arange(shape[0]), targets.flatten()] = 1
            if mode == 'concatenate':
                shape = one_hot.shape[-3:-2] + (reduce(mul,
                                                       one_hot.shape[-2:], 1),)
                one_hot = one_hot.reshape(shape)
            elif mode == 'merge':
                one_hot = np.minimum(one_hot.sum(axis=one_hot.ndim - 2), 1)
        return one_hot

    def theano_expr(self, targets, mode='stack', sparse=False):
        """
        Return the one-hot transformation as a symbolic expression.
        If labels appear multiple times, their value in the one-hot
        vector is incremented.

        Parameters
        ----------
        targets : tensor_like, 1- or 2-dimensional, integer dtype
            A symbolic tensor representing labels as integers
            between 0 and `max_labels` - 1, `max_labels` supplied
            at formatter construction.
        mode : string
            The way in which to convert the labels to arrays. Takes
            three different options:

              - "concatenate" : concatenates the one-hot vectors from
                multiple labels
              - "stack" : returns a matrix where each row is the
                one-hot vector of a label
              - "merge" : merges the one-hot vectors together to
                form a vector where the elements are
                the result of an indicator function
                NB: As the result of an indicator function
                the result is the same in case a label
                is duplicated in the input.
        sparse : bool
            If true then the return value is sparse matrix. Note that
            if sparse is True, then mode cannot be 'stack' because
            sparse matrices need to be 2D

        Returns
        -------
        one_hot : TensorVariable, 1, 2 or 3-dimensional, sparse or dense
            A symbolic tensor representing a one-hot encoding of the
            supplied labels.
        """
        if mode not in ('concatenate', 'stack', 'merge'):
            raise ValueError("%s got bad mode argument '%s'" %
                             (self.__class__.__name__, str(self._max_labels)))
        elif mode == 'stack' and sparse:
            raise ValueError("Sparse matrices need to be 2D, hence they"
                             "cannot be stacked")
        squeeze_required = False
        if targets.ndim != 2:
            if targets.ndim == 1:
                squeeze_required = True
                targets = targets.dimshuffle('x', 0)
            else:
                raise ValueError("targets tensor must be 1 or 2-dimensional")
        if 'int' not in str(targets.dtype):
            raise TypeError("need an integer tensor for targets")
        if sparse:
            if mode == 'concatenate':
                one_hot = theano.sparse.CSR(
                    tensor.ones_like(targets, dtype=self._dtype).flatten(),
                    (targets.flatten() + tensor.arange(targets.size) *
                     self._max_labels) % (self._max_labels * targets.shape[1]),
                    tensor.arange(targets.shape[0] + 1) * targets.shape[1],
                    tensor.stack(targets.shape[0],
                                 self._max_labels * targets.shape[1])
                )
            else:
                one_hot = theano.sparse.CSR(
                    tensor.ones_like(targets, dtype=self._dtype).flatten(),
                    targets.flatten(),
                    tensor.arange(targets.shape[0] + 1) * targets.shape[1],
                    tensor.stack(targets.shape[0], self._max_labels)
                )
        else:
            if mode == 'concatenate':
                one_hot = tensor.zeros((targets.shape[0] * targets.shape[1],
                                        self._max_labels), dtype=self._dtype)
                one_hot = tensor.set_subtensor(
                    one_hot[tensor.arange(targets.size),
                            targets.flatten()], 1)
                one_hot = one_hot.reshape(
                    (targets.shape[0], targets.shape[1] * self._max_labels)
                )
            elif mode == 'merge':
                one_hot = tensor.zeros((targets.shape[0], self._max_labels),
                                       dtype=self._dtype)
                one_hot = tensor.set_subtensor(
                    one_hot[tensor.arange(targets.size) % targets.shape[0],
                            targets.T.flatten()], 1)
            else:
                one_hot = tensor.zeros((targets.shape[0], targets.shape[1],
                                        self._max_labels), dtype=self._dtype)
                one_hot = tensor.set_subtensor(one_hot[
                    tensor.arange(targets.shape[0]).reshape((targets.shape[0],
                                                             1)),
                    tensor.arange(targets.shape[1]),
                    targets
                ], 1)
            if squeeze_required:
                if one_hot.ndim == 2:
                    one_hot = one_hot.reshape((one_hot.shape[1],))
                if one_hot.ndim == 3:
                    one_hot = one_hot.reshape((one_hot.shape[1],
                                               one_hot.shape[2]))
        return one_hot


def convert_to_one_hot(integer_vector, dtype=None, max_labels=None,
                       mode='stack', sparse=False):
    """
    Formats a given array of target labels into a one-hot
    vector.

    Parameters
    ----------
    max_labels : int, optional
        The number of possible classes/labels. This means that
        all labels should be < max_labels. Example: For MNIST
        there are 10 numbers and hence max_labels = 10. If not
        given it defaults to max(integer_vector) + 1.
    dtype : dtype, optional
        The desired dtype for the converted one-hot vectors.
        Defaults to config.floatX if not given.
    integer_vector : ndarray
        A 1D array of targets, or a batch (2D array) where
        each row is a list of targets.
    mode : string
        The way in which to convert the labels to arrays. Takes
        three different options:

          - "concatenate" : concatenates the one-hot vectors from
            multiple labels
          - "stack" : returns a matrix where each row is the
            one-hot vector of a label
          - "merge" : merges the one-hot vectors together to
            form a vector where the elements are
            the result of an indicator function
    sparse : bool
        If true then the return value is sparse matrix. Note that
        if sparse is True, then mode cannot be 'stack' because
        sparse matrices need to be 2D

    Returns
    -------
    one_hot : NumPy array
       Can be 1D-3D depending on settings. Normally, the first axis are
       the different batch items, the second axis the labels, the third
       axis the one_hot vectors. Can be dense or sparse.
    """
    if dtype is None:
        dtype = config.floatX
    if isinstance(integer_vector, list):
        integer_vector = np.array(integer_vector)
    assert np.min(integer_vector) >= 0
    assert integer_vector.ndim <= 2
    if max_labels is None:
        max_labels = max(integer_vector) + 1
    return OneHotFormatter(max_labels, dtype=dtype).format(
        integer_vector, mode=mode, sparse=sparse
    )


def _validate_labels(labels, ndim):
    """
    Validate that the passed label is in a right data type, and convert
    it into the desired shape.

    Parameters
    ----------
    labels : array_like, 1-dimensional (or 2-dimensional (nlabels, 1))
        The integer labels to use to construct the one hot matrix.
    ndim : int
        Number of dimensions the label have.

    Returns
    -------
    labels : ndarray, (nlabels, ) or (nlabels, )
        The resulting label vector.
    """
    labels = np.asarray(labels)
    if labels.dtype.kind not in ('u', 'i'):
        raise ValueError("labels must have int or uint dtype")
    if ndim == 1 and labels.ndim != 1:
        if labels.ndim == 2 and labels.shape[1] == 1:
            labels = labels.squeeze()
        else:
            raise ValueError("labels must be 1-dimensional")
    elif ndim == 2 and labels.ndim != 2:
        raise ValueError("labels must be 2-dimensional, no ragged "
                         "lists-of-lists")
    return labels


def compressed_one_hot(labels, dtype=None, out=None, simplify_binary=True,
                       mode='stack', sparse=False):
    """
    Construct a one-hot matrix from a vector of integer labels, but
    only including columns corresponding to integer labels that
    actually appear.

    Parameters
    ----------
    labels : array_like, 1-dimensional (or 2-dimensional (nlabels, 1))
        The integer labels to use to construct the one hot matrix.

    dtype : str or dtype object, optional
        The dtype you wish the returned array to have. Defaults
        to `labels.dtype` if not provided.

    out : ndarray, optional
        An array to use in lieu of allocating one. Must be the
        right shape, i.e. same first dimension as `labels` and
        second dimension greater than or equal to the number of
        unique values in `labels`.

    simplify_binary : bool, optional
        If `True`, if there are only two distinct labels, return
        an `(nlabels, 1)` matrix with 0 denoting the lesser integer
        label and 1 denoting the greater, instead of a redundant
        `(nlabels, 2)` matrix.
    mode : string
        The way in which to convert the labels to arrays. Takes
        three different options:

            - "concatenate" : concatenates the one-hot vectors from
              multiple labels
            - "stack" : returns a matrix where each row is the
              one-hot vector of a label
            - "merge" : merges the one-hot vectors together to
              form a vector where the elements are
              the result of an indicator function
              NB: As the result of an indicator function
              the result is the same in case a label
              is duplicated in the input.
    sparse : bool
        If true then the return value is sparse matrix. Note that
        if sparse is True, then mode cannot be 'stack' because
        sparse matrices need to be 2D

    Returns
    -------
    out : ndarray, (nlabels, max_label + 1) or (nlabels, 1)
        The resulting one-hot matrix.

    uniq : ndarray, 1-dimensional
        The array of unique values in `labels` in the order
        in which the corresponding columns appear in `out`.
    """
    labels = _validate_labels(labels, ndim=1)
    labels_ = labels.copy()
    uniq = np.unique(labels_)
    for i, e in enumerate(uniq):
        labels_[labels_ == e] = i
    if simplify_binary and len(uniq) == 2:
        return labels_.reshape((labels_.shape[0], 1)), uniq
    else:
        return OneHotFormatter(len(uniq), dtype=dtype).format(
            labels_, mode=mode, sparse=sparse), uniq
