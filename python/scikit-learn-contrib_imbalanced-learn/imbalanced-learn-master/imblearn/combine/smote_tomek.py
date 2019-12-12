"""Class to perform over-sampling using SMOTE and cleaning using Tomek
links."""
from __future__ import division, print_function

import warnings

from ..base import BaseBinarySampler
from ..over_sampling import SMOTE
from ..under_sampling import TomekLinks


class SMOTETomek(BaseBinarySampler):
    """Class to perform over-sampling using SMOTE and cleaning using
    Tomek links.

    Combine over- and under-sampling using SMOTE and Tomek links.

    Parameters
    ----------
    ratio : str or float, optional (default='auto')
        If 'auto', the ratio will be defined automatically to balance
        the dataset. Otherwise, the ratio is defined as the
        number of samples in the minority class over the the number of
        samples in the majority class.

    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by np.random.

    smote : object, optional (default=SMOTE())
        The SMOTE object to use. If not given, a SMOTE object with default
        parameters will be given.

    tomek : object, optional (default=Tomek())
        The Tomek object to use. If not given, a Tomek object with default
        parameters will be given.

    k : int, optional (default=None)
        Number of nearest neighbours to used to construct synthetic
        samples.

        NOTE: `k` is deprecated from 0.2 and will be replaced in 0.4
        Give directly a SMOTE object.

    m : int, optional (default=None)
        Number of nearest neighbours to use to determine if a minority
        sample is in danger.

        NOTE: `m` is deprecated from 0.2 and will be replaced in 0.4
        Give directly a SMOTE object.

    out_step : float, optional (default=None)
        Step size when extrapolating.

        NOTE: `out_step` is deprecated from 0.2 and will be replaced in 0.4
        Give directly a SMOTE object.

    kind_smote : str, optional (default=None)
        The type of SMOTE algorithm to use one of the following
        options: 'regular', 'borderline1', 'borderline2', 'svm'.

    NOTE: `kind_smote` is deprecated from 0.2 and will be replaced in 0.4
        Give directly a SMOTE object.

    n_jobs : int, optional (default=None)
        The number of threads to open if possible.

        NOTE: `n_jobs` is deprecated from 0.2 and will be replaced in 0.4
        Give directly a SMOTE object.

    Attributes
    ----------
    min_c_ : str or int
        The identifier of the minority class.

    max_c_ : str or int
        The identifier of the majority class.

    stats_c_ : dict of str/int : int
        A dictionary in which the number of occurences of each class is
        reported.

    X_shape_ : tuple of int
        Shape of the data `X` during fitting.

    Notes
    -----
    The methos is presented in [1]_.

    This class does not support mutli-class.

    Examples
    --------

    >>> from collections import Counter
    >>> from sklearn.datasets import make_classification
    >>> from imblearn.combine import \
    SMOTETomek # doctest: +NORMALIZE_WHITESPACE
    >>> X, y = make_classification(n_classes=2, class_sep=2,
    ... weights=[0.1, 0.9], n_informative=3, n_redundant=1, flip_y=0,
    ... n_features=20, n_clusters_per_class=1, n_samples=1000, random_state=10)
    >>> print('Original dataset shape {}'.format(Counter(y)))
    Original dataset shape Counter({1: 900, 0: 100})
    >>> smt = SMOTETomek(random_state=42)
    >>> X_res, y_res = smt.fit_sample(X, y)
    >>> print('Resampled dataset shape {}'.format(Counter(y_res)))
    Resampled dataset shape Counter({0: 900, 1: 900})

    References
    ----------
    .. [1] G. Batista, B. Bazzan, M. Monard, "Balancing Training Data for
       Automated Annotation of Keywords: a Case Study," In WOB, 10-18, 2003.

    """

    def __init__(self,
                 ratio='auto',
                 random_state=None,
                 smote=None,
                 tomek=None,
                 k=None,
                 m=None,
                 out_step=None,
                 kind_smote=None,
                 n_jobs=None):
        super(SMOTETomek, self).__init__(
            ratio=ratio, random_state=random_state)
        self.smote = smote
        self.tomek = tomek
        self.k = k
        self.m = m
        self.out_step = out_step
        self.kind_smote = kind_smote
        self.n_jobs = n_jobs

    def _validate_estimator(self):
        "Private function to validate SMOTE and ENN objects"

        # Check any parameters for SMOTE was provided
        # Anounce deprecation
        if (self.k is not None or self.m is not None or
                self.out_step is not None or self.kind_smote is not None or
                self.n_jobs is not None):
            warnings.warn('Parameters initialization will be replaced in'
                          ' version 0.4. Use a SMOTE object instead.',
                          DeprecationWarning)
            # We need to list each parameter and decide if we affect a default
            # value or not
            if self.k is None:
                self.k = 5
            if self.m is None:
                self.m = 10
            if self.out_step is None:
                self.out_step = 0.5
            if self.kind_smote is None:
                self.kind_smote = 'regular'
            if self.n_jobs is None:
                smote_jobs = 1
            else:
                smote_jobs = self.n_jobs
            self.smote_ = SMOTE(
                ratio=self.ratio,
                random_state=self.random_state,
                k=self.k,
                m=self.m,
                out_step=self.out_step,
                kind=self.kind_smote,
                n_jobs=smote_jobs)
        # If an object was given, affect
        elif self.smote is not None:
            if isinstance(self.smote, SMOTE):
                self.smote_ = self.smote
            else:
                raise ValueError('smote needs to be a SMOTE object.')
        # Otherwise create a default SMOTE
        else:
            self.smote_ = SMOTE(
                ratio=self.ratio, random_state=self.random_state)

        # Check any parameters for ENN was provided
        # Anounce deprecation
        if self.n_jobs is not None:
            warnings.warn('Parameters initialization will be replaced in'
                          ' version 0.4. Use a ENN object instead.',
                          DeprecationWarning)
            self.tomek_ = TomekLinks(
                random_state=self.random_state, n_jobs=self.n_jobs)
        # If an object was given, affect
        elif self.tomek is not None:
            if isinstance(self.tomek, TomekLinks):
                self.tomek_ = self.tomek
            else:
                raise ValueError('tomek needs to be a TomekLinks object.')
        # Otherwise create a default EditedNearestNeighbours
        else:
            self.tomek_ = TomekLinks(random_state=self.random_state)

    def fit(self, X, y):
        """Find the classes statistics before to perform sampling.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            Matrix containing the data which have to be sampled.

        y : ndarray, shape (n_samples, )
            Corresponding label for each sample in X.

        Returns
        -------
        self : object,
            Return self.

        """

        super(SMOTETomek, self).fit(X, y)

        self._validate_estimator()

        # Fit using SMOTE
        self.smote_.fit(X, y)

        return self

    def _sample(self, X, y):
        """Resample the dataset.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            Matrix containing the data which have to be sampled.

        y : ndarray, shape (n_samples, )
            Corresponding label for each sample in X.

        Returns
        -------
        X_resampled : ndarray, shape (n_samples_new, n_features)
            The array containing the resampled data.

        y_resampled : ndarray, shape (n_samples_new)
            The corresponding label of `X_resampled`

        """

        # Transform using SMOTE
        X, y = self.smote_.sample(X, y)

        # Fit and transform using ENN
        return self.tomek_.fit_sample(X, y)
