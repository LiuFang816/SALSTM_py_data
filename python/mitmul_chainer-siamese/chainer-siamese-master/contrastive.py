import numpy

from chainer import cuda
from chainer import function
from chainer.utils import type_check


class Contrastive(function.Function):

    """Contrastive loss function."""

    def __init__(self, margin, use_cudnn=True):
        self.margin = float(margin)
        self.use_cudnn = use_cudnn

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 3)

        x0_type, x1_type, y_type = in_types
        type_check.expect(
            x0_type.dtype == numpy.float32,
            x1_type.dtype == numpy.float32,
            x0_type.shape == x1_type.shape,
            x0_type.shape[0] == x1_type.shape[0],
            x1_type.shape[0] == y_type.shape[0],
            x0_type.ndim == 2,
            x1_type.ndim == 2,
            y_type.ndim == 1
        )

    def forward(self, inputs):
        xp = cuda.get_array_module(*inputs)
        x0, x1, y = inputs

        self.diff = x0 - x1  # N x 2
        self.dist_sq = xp.sum(self.diff ** 2, axis=1)  # N
        self.dist = xp.sqrt(self.dist_sq)
        self.mdist = self.margin - self.dist
        dist = xp.maximum(self.mdist, 0)
        loss = y * self.dist_sq + (1 - y) * dist * dist
        loss = xp.sum(loss) / 2.0 / x0.shape[0]

        return xp.array(loss, dtype=xp.float32),

    def backward(self, inputs, gy):
        xp = cuda.get_array_module(*inputs)
        x0, x1, y = inputs

        y = xp.vstack((y, y)).T
        alpha = gy[0] / y.shape[0]
        dist = xp.vstack((self.dist, self.dist)).T
        # similar pair
        gx0 = alpha * y * self.diff
        # dissimilar pair
        mdist = xp.vstack((self.mdist, self.mdist)).T
        mdist_p = xp.array(self.mdist > 0, dtype=xp.int32)
        mdist_p = xp.vstack((mdist_p, mdist_p)).T
        gx0 += alpha * (1 - y) * mdist_p * mdist * -(self.diff / dist)
        gx0 = gx0.astype(xp.float32)

        return gx0, -gx0, None


def contrastive(x0, x1, y, margin=1, use_cudnn=True):
    """Contrastive loss.

    """
    return Contrastive(margin, use_cudnn)(x0, x1, y)
