import numpy as np
import gym
from gym.spaces import prng
import torch

class MultipleSpaces(gym.Space):
    '''
    Defines a space as a tuple of sensors spaces
    '''
    def __init__(self,*spaces):
        self.spaces=spaces

    def sample(self):
        r=None
        for s in self.spaces:
            if (r==None):
                r=(s,)
            else:
                r=r+(s,)
        return r

    def contains(self,x):
        pos=0
        for s in self.spaces:
            if (not s.contains(x[pos])):
                return False
            pos=pos+1
        return True


    def to_jsonable(self, sample_n):
        raise NonImplementedError

    def from_jsonable(self, sample_n):
        raise NonImplementedError


    @property
    def __eq__(self, other):
        if (len(self.spaces)!=len(other.spaces)):
            return False
        pos=0
        for s in self.spaces:
            if (not s.__eq__(other.spaces[pos])):
                return False
            pos=pos+1


class PytorchBox(gym.Space):
    """
    A box in R^n
    I.e., each coordinate is bounded.
    """

    def __init__(self, low, high, shape=None):
        """
        Two kinds of valid input:
            Box(-1.0, 1.0, (3,4)) # low and high are scalars, and shape is provided
            Box(np.array([-1.0,-2.0]), np.array([2.0,4.0])) # low and high are arrays of the same shape
        """
        if shape is None:
            assert low.shape == high.shape
            self.low = low
            self.high = high
        else:
            assert np.isscalar(low) and np.isscalar(high)
            self.low = low + np.zeros(shape)
            self.high = high + np.zeros(shape)
        self.ndimensions = len(self.low.shape)

    def sample(self):
        return torch.Tensor(prng.np_random.uniform(low=self.low, high=self.high, size=self.low.shape))

    def contains(self, xx):
        x = xx.numpy()
        return x.shape == self.shape and (x >= self.low).all() and (x <= self.high).all()

    def to_jsonable(self, sample_n):
        raise NonImplementedError

    def from_jsonable(self, sample_n):
        raise NonImplementedError

    @property
    def __eq__(self, other):
        return np.allclose(self.low, other.low) and np.allclose(self.high, other.high)


class CheapPytorchBox(gym.Space):
    """
    A box in R^n. Coordinates are not bounded.
    """

    def __init__(self,shape):
        self.low=np.zeros(shape)
        self.high=np.zeros(shape)
        self.ndimensions=len(self.low.shape)
        pass

    def sample(self):
        raise NotImplementedError

    def contains(self, xx):
        raise NotImplementedError

    def to_jsonable(self, sample_n):
        raise NonImplementedError

    def from_jsonable(self, sample_n):
        raise NonImplementedError

    @property
    def __eq__(self, other):
        return True