import numpy as np
import theano.tensor as T

def increment_odd(x):
    """
    x: a Theano vector
    Returns:
    y: a Theano vector equal to x, but with all odd-numbered elements
    incremented by 1.
    """

    y = T.inc_subtensor(x[1::2], 1.)
    return y


if __name__ == "__main__":
    x = T.vector()
    xv = np.zeros((4,), dtype=x.dtype)
    yv = increment_odd(x).eval({x:xv})
    assert np.allclose(yv, np.array([0., 1., 0., 1.]))
