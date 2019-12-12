"""K-means as a postprocessing Block subclass."""

import logging
import numpy
from theano.compat.six.moves import xrange
from pylearn2.blocks import Block
from pylearn2.models.model import Model
from pylearn2.space import VectorSpace
from pylearn2.utils import sharedX
from pylearn2.utils.mem import improve_memory_error_message
from pylearn2.utils import wraps
from pylearn2.utils import contains_nan
import warnings

try:
    import milk
except ImportError:
    milk = None
    warnings.warn(""" Install milk ( http://packages.python.org/milk/ )
                    It has a better k-means implementation. Falling back to
                    our own really slow implementation. """)

logger = logging.getLogger(__name__)


class KMeans(Block, Model):
    """
    Block that outputs a vector of probabilities that a sample belong
    to means computed during training.

    Parameters
    ----------
    k : int
        Number of clusters
    nvis : int
        Dimension of input
    convergence_th : float, optional
        Threshold of distance to clusters under which k-means stops
        iterating.
    max_iter : int, optional
        Maximum number of iterations. Defaults to infinity.
    verbose : bool
        WRITEME
    """

    def __init__(self, k, nvis, convergence_th=1e-6, max_iter=None,
                 verbose=False):
        Block.__init__(self)
        Model.__init__(self)

        self.input_space = VectorSpace(nvis)

        self.k = k
        self.mu = None
        self.convergence_th = convergence_th
        if max_iter:
            if max_iter < 0:
                raise Exception('KMeans init: max_iter should be positive.')
            self.max_iter = max_iter
        else:
            self.max_iter = float('inf')

        self.verbose = verbose

    def train_all(self, dataset, mu=None):
        """
        Process kmeans algorithm on the input to localize clusters.

        Parameters
        ----------
        dataset : WRITEME
        mu : WRITEME

        Returns
        -------
        rval : bool
            WRITEME
        """

        # TODO-- why does this sometimes return X and sometimes return nothing?

        X = dataset.get_design_matrix()

        n, m = X.shape
        k = self.k

        if milk is not None:
            # use the milk implementation of k-means if it's available
            cluster_ids, mu = milk.kmeans(X, k)
        else:
            # our own implementation

            # taking random inputs as initial clusters if user does not provide
            # them.
            if mu is not None:
                if not len(mu) == k:
                    raise Exception("You gave %i clusters"
                                    ", but k=%i were expected"
                                    % (len(mu), k))
            else:
                indices = numpy.random.randint(X.shape[0], size=k)
                mu = X[indices]

            try:
                dists = numpy.zeros((n, k))
            except MemoryError as e:
                improve_memory_error_message(e, "dying trying to allocate "
                                                "dists matrix for {0} "
                                                "examples and {1} "
                                                "means".format(n, k))

            old_kills = {}

            iter = 0
            mmd = prev_mmd = float('inf')
            while True:
                if self.verbose:
                    logger.info('kmeans iter {0}'.format(iter))

                # print 'iter:',iter,' conv crit:',abs(mmd-prev_mmd)
                # if numpy.sum(numpy.isnan(mu)) > 0:
                if contains_nan(mu):
                    logger.info('nan found')
                    return X

                # computing distances
                for i in xrange(k):
                    dists[:, i] = numpy.square((X - mu[i, :])).sum(axis=1)

                if iter > 0:
                    prev_mmd = mmd

                min_dists = dists.min(axis=1)

                # mean minimum distance:
                mmd = min_dists.mean()

                logger.info('cost: {0}'.format(mmd))

                if iter > 0 and (iter >= self.max_iter or
                                 abs(mmd - prev_mmd) < self.convergence_th):
                    # converged
                    break

                # finding minimum distances
                min_dist_inds = dists.argmin(axis=1)

                # computing means
                i = 0
                blacklist = []
                new_kills = {}
                while i < k:
                    b = min_dist_inds == i
                    if not numpy.any(b):
                        killed_on_prev_iter = True
                        # initializes empty cluster to be the mean of the d
                        # data points farthest from their corresponding means
                        if i in old_kills:
                            d = old_kills[i] - 1
                            if d == 0:
                                d = 50
                            new_kills[i] = d
                        else:
                            d = 5
                        mu[i, :] = 0
                        for j in xrange(d):
                            idx = numpy.argmax(min_dists)
                            min_dists[idx] = 0
                            # chose point idx
                            mu[i, :] += X[idx, :]
                            blacklist.append(idx)
                        mu[i, :] /= float(d)
                        # cluster i was empty, reset it to d far out data
                        # points recomputing distances for this cluster
                        dists[:, i] = numpy.square((X - mu[i, :])).sum(axis=1)
                        min_dists = dists.min(axis=1)
                        for idx in blacklist:
                            min_dists[idx] = 0
                        min_dist_inds = dists.argmin(axis=1)
                        # done
                        i += 1
                    else:
                        mu[i, :] = numpy.mean(X[b, :], axis=0)
                        if contains_nan(mu):
                            logger.info('nan found at {0}'.format(i))
                            return X
                        i += 1

                old_kills = new_kills

                iter += 1

        self.mu = sharedX(mu)
        self._params = [self.mu]

    @wraps(Model.continue_learning)
    def continue_learning(self):
        # One call to train_all currently trains the model fully,
        # so return False immediately.
        return False

    def get_params(self):
        """
        .. todo::

            WRITEME
        """
        # patch older pkls
        if not hasattr(self.mu, 'get_value'):
            self.mu = sharedX(self.mu)
        if not hasattr(self, '_params'):
            self._params = [self.mu]

        return [param for param in self._params]

    def __call__(self, X):
        """
        Compute for each sample its probability to belong to a cluster.

        Parameters
        ----------
        X : numpy.ndarray
            Matrix of sampless of shape (n, d)

        Returns
        -------
        WRITEME
        """
        n, m = X.shape
        k = self.k
        mu = self.mu
        dists = numpy.zeros((n, k))
        for i in xrange(k):
            dists[:, i] = numpy.square((X - mu[i, :])).sum(axis=1)
        return dists / dists.sum(axis=1).reshape(-1, 1)

    def get_weights(self):
        """
        .. todo::

            WRITEME
        """
        return self.mu

    def get_weights_format(self):
        """
        .. todo::

            WRITEME
        """
        return ['h', 'v']

    # Use version defined in Model, rather than Block (which raises
    # NotImplementedError).
    get_input_space = Model.get_input_space
    get_output_space = Model.get_output_space
