"""Convolution Neural Network example with both MinPy ndarray and MXNet symbol."""
import sys
import argparse

import minpy
import minpy.numpy as np
import numpy
import mxnet as mx
from minpy.nn.io import NDArrayIter
# Can also use MXNet IO here
# from mxnet.io import NDArrayIter
from minpy.core import Function
from minpy.nn import layers
from minpy.nn.model import ModelBase
from minpy.nn.solver import Solver
from examples.utils.data_utils import get_CIFAR10_data
from minpy.primitive import customop

# Please uncomment following if you have GPU-enabled MXNet installed.
# from minpy.context import set_context, gpu
# set_context(gpu(0))  # set the global context as gpu(0)

batch_size = 128
input_size = (3, 32, 32)
flattened_input_size = 3 * 32 * 32
hidden_size = 512
num_classes = 10


@customop('numpy')
def my_softmax(x, y):
    probs = numpy.exp(x - numpy.max(x, axis=1, keepdims=True))
    probs /= numpy.sum(probs, axis=1, keepdims=True)
    N = x.shape[0]
    loss = -numpy.sum(numpy.log(probs[numpy.arange(N), y])) / N
    return loss


def my_softmax_grad(ans, x, y):
    def grad(g):
        N = x.shape[0]
        probs = numpy.exp(x - numpy.max(x, axis=1, keepdims=True))
        probs /= numpy.sum(probs, axis=1, keepdims=True)
        probs[numpy.arange(N), y] -= 1
        probs /= N
        return probs

    return grad

my_softmax.def_grad(my_softmax_grad)


class ConvolutionNet(ModelBase):
    def __init__(self):
        super(ConvolutionNet, self).__init__()
        # Define symbols that using convolution and max pooling to extract better features
        # from input image.
        net = mx.sym.Variable(name='X')
        net = mx.sym.Convolution(
            data=net, name='conv', kernel=(7, 7), num_filter=32)
        net = mx.sym.Activation(data=net, act_type='relu')
        net = mx.sym.Pooling(
            data=net,
            name='pool',
            pool_type='max',
            kernel=(2, 2),
            stride=(2, 2))
        net = mx.sym.Flatten(data=net)
        # Create forward function and add parameters to this model.
        self.conv = Function(
            net, input_shapes={'X': (batch_size, ) + input_size}, name='conv')
        self.add_params(self.conv.get_params())
        # Define ndarray parameters used for classification part.
        output_shape = self.conv.get_one_output_shape()
        conv_out_size = output_shape[1]
        self.add_param(name='w1', shape=(conv_out_size, hidden_size)) \
            .add_param(name='b1', shape=(hidden_size,)) \
            .add_param(name='w2', shape=(hidden_size, num_classes)) \
            .add_param(name='b2', shape=(num_classes,))

    def forward(self, X, mode):
        out = self.conv(X=X, **self.params)
        out = layers.affine(out, self.params['w1'], self.params['b1'])
        out = layers.relu(out)
        out = layers.affine(out, self.params['w2'], self.params['b2'])
        return out

    def loss(self, predict, y):
        return my_softmax(predict, y)


def main(args):
    # Create model.
    model = ConvolutionNet()
    # Create data iterators for training and testing sets.
    data = get_CIFAR10_data(args.data_dir)
    train_dataiter = NDArrayIter(
        data=data['X_train'],
        label=data['y_train'],
        batch_size=batch_size,
        shuffle=True)
    test_dataiter = NDArrayIter(
        data=data['X_test'],
        label=data['y_test'],
        batch_size=batch_size,
        shuffle=False)
    # Create solver.
    solver = Solver(
        model,
        train_dataiter,
        test_dataiter,
        num_epochs=10,
        init_rule='gaussian',
        init_config={'stdvar': 0.001},
        update_rule='sgd_momentum',
        optim_config={'learning_rate': 1e-3,
                      'momentum': 0.9},
        verbose=True,
        print_every=20)
    # Initialize model parameters.
    solver.init()
    # Train!
    solver.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Multi-layer perceptron example using minpy operators")
    parser.add_argument(
        '--data_dir',
        type=str,
        required=True,
        help='Directory that contains cifar10 data')
    main(parser.parse_args())
