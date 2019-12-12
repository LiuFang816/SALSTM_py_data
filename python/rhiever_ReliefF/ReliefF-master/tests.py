"""
    Unit tests for ReliefF.
"""

from ReliefF import ReliefF
import numpy as np
import timeit
from sklearn.cross_validation import train_test_split


def test_init():
    """Make sure ReliefF instantiates correctly"""
    fs = ReliefF(n_neighbors=50, n_features_to_keep=100)
    assert fs.n_neighbors == 50
    assert fs.n_features_to_keep == 100


def test_fit():
    """Make sure ReliefF fits correctly"""
    X_train, X_test, y_train, y_test = get_testing_data()

    fs = ReliefF(n_neighbors=100, n_features_to_keep=5)
    fs.fit(X_train, y_train)

    with np.load("data/test_arrays.npz") as arrays:
        correct_top_features = arrays['correct_top_features']
        correct_feature_scores = arrays['correct_feature_scores']

    assert np.all(np.equal(fs.top_features, correct_top_features))
    assert np.all(np.equal(fs.feature_scores, correct_feature_scores))


def test_transform():
    """Make sure ReliefF transforms correctly"""
    X_train, X_test, y_train, y_test = get_testing_data()

    fs = ReliefF(n_neighbors=100, n_features_to_keep=5)
    fs.fit(X_train, y_train)
    X_test = fs.transform(X_test)

    assert np.all(np.equal(X_test[0], np.array([0, 1, 1, 1, 1])))
    assert np.all(np.equal(X_test[1], np.array([2, 1, 0, 1, 1])))
    assert np.all(np.equal(X_test[-2], np.array([1, 1, 0, 1, 0])))
    assert np.all(np.equal(X_test[-1], np.array([1, 0, 1, 0, 0])))


def test_fit_transform():
    """Make sure ReliefF fit_transforms correctly"""
    X_train, X_test, y_train, y_test = get_testing_data()

    fs = ReliefF(n_neighbors=100, n_features_to_keep=5)
    X_train = fs.fit_transform(X_train, y_train)

    assert np.all(np.equal(X_train[0], np.array([1, 1, 0, 2, 1])))
    assert np.all(np.equal(X_train[1], np.array([0, 0, 0, 2, 0])))
    assert np.all(np.equal(X_train[-2], np.array([1, 1, 0, 1, 0])))
    assert np.all(np.equal(X_train[-1], np.array([0, 0, 0, 0, 0])))


def time_fit(n_neighbors=100, n_features_to_keep=5, r=3, n=10):
    """Time the execution of ReliefF.fit(), 10 times by default"""
    setup = ';'.join(["from ReliefF import ReliefF",
                      "from tests import get_testing_data",
                      "X_train, X_test, y_train, y_test = get_testing_data()",
                      "fs = ReliefF(n_neighbors=%d, n_features_to_keep=%d)"
                      % (n_neighbors, n_features_to_keep)])
    fit = "fs.fit(X_train, y_train)"

    # Should print out r measurements (in seconds) to execute fit() n times.
    print(timeit.repeat(stmt=fit, setup=setup, repeat=r, number=n))


def get_testing_data(file='data/GAMETES-test.csv.gz', target=-1,
                     return_header=False, head=0):
    """Loads the desired data for testing ReliefF's functions"""
    data = np.genfromtxt(file, delimiter=',')
    header, data = data[head], data[head+1:]  # Separate header (first) row
    y = data[:, target]  # labels of the data
    X = np.delete(data, target, axis=1)  # Remove labels from data

    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        random_state=34895)

    # The user can decide if they want the headers returned as first value
    if return_header:
        return header, X_train, X_test, y_train, y_test

    return X_train, X_test, y_train, y_test
