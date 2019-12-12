import numpy
import theano
from numpy.testing import assert_allclose, assert_raises_regex
from theano import tensor

from blocks import bricks
from blocks.bricks.base import application
from blocks.graph import ComputationGraph
from blocks.monitoring.aggregation import (mean, Mean, Minimum, Maximum,
                                           Concatenate)
from blocks.utils import shared_floatx

from collections import OrderedDict
from fuel.datasets import IndexableDataset
from fuel.streams import DataStream
from fuel.schemes import SequentialScheme

from blocks.monitoring.evaluators import DatasetEvaluator, AggregationBuffer


class TestBrick(bricks.Brick):
    def _allocate(self):
        self.parameters = [shared_floatx(2, name='V')]

    @application(inputs=['input_'], outputs=['output'])
    def apply(self, input_, application_call):
        V = self.parameters[0]
        mean_row_mean = mean(input_.mean(axis=1).sum(), input_.shape[0])
        application_call.add_auxiliary_variable((V ** 2).sum(),
                                                name='V_squared')
        application_call.add_auxiliary_variable(mean_row_mean,
                                                name='mean_row_mean')
        application_call.add_auxiliary_variable(input_.mean(),
                                                name='mean_batch_element')
        return input_ + V


def test_parameter_monitor():
    X = tensor.matrix('X')
    brick = TestBrick(name='test_brick')
    y = brick.apply(X)
    graph = ComputationGraph([y])

    # Test the monitors without aggregation schemes
    monitors = [v for v in graph.auxiliary_variables
                if not hasattr(v.tag, 'aggregation_scheme')]
    monitors.sort(key=lambda variable: variable.name)

    f = theano.function([X], monitors)
    monitor_vals = f(numpy.arange(4, dtype=theano.config.floatX).reshape(2, 2))
    assert_allclose(monitor_vals, [4., 1.5])

    # Test the aggregation scheme
    monitor, = [v for v in graph.auxiliary_variables
                if hasattr(v.tag, 'aggregation_scheme')]
    aggregator = monitor.tag.aggregation_scheme.get_aggregator()
    initialize = theano.function([], updates=aggregator.initialization_updates)
    initialize()
    aggregate = theano.function([X], updates=aggregator.accumulation_updates)
    aggregate(numpy.arange(4, dtype=theano.config.floatX).reshape(2, 2))
    aggregate(numpy.arange(4, 10, dtype=theano.config.floatX).reshape(3, 2))
    assert_allclose(aggregator.readout_variable.eval(), 4.5)


def test_mean_aggregator():
    num_examples = 4
    batch_size = 2

    features = numpy.array([[0, 3],
                           [2, 9],
                           [2, 4],
                           [5, 1]], dtype=theano.config.floatX)

    dataset = IndexableDataset(OrderedDict([('features', features)]))

    data_stream = DataStream(dataset,
                             iteration_scheme=SequentialScheme(num_examples,
                                                               batch_size))

    x = tensor.matrix('features')
    y = (x**2).mean(axis=0)
    y.name = 'y'
    z = y.sum()
    z.name = 'z'

    y.tag.aggregation_scheme = Mean(y, 1.)
    z.tag.aggregation_scheme = Mean(z, 1.)

    assert_allclose(DatasetEvaluator([y]).evaluate(data_stream)['y'],
                    numpy.array([8.25, 26.75], dtype=theano.config.floatX))
    assert_allclose(DatasetEvaluator([z]).evaluate(data_stream)['z'],
                    numpy.array([35], dtype=theano.config.floatX))


def test_min_max_aggregators():
    num_examples = 4
    batch_size = 2

    features = numpy.array([[2, 3],
                           [2, 9],
                           [2, 4],
                           [5, 1]], dtype=theano.config.floatX)

    dataset = IndexableDataset(OrderedDict([('features', features)]))

    data_stream = DataStream(dataset,
                             iteration_scheme=SequentialScheme(num_examples,
                                                               batch_size))

    x = tensor.matrix('features')
    y = (x**2).sum(axis=0)
    y.name = 'y'
    z = y.min()
    z.name = 'z'

    y.tag.aggregation_scheme = Maximum(y)
    z.tag.aggregation_scheme = Minimum(z)

    assert_allclose(DatasetEvaluator([y]).evaluate(data_stream)['y'],
                    numpy.array([29, 90], dtype=theano.config.floatX))
    assert_allclose(DatasetEvaluator([z]).evaluate(data_stream)['z'],
                    numpy.array([8], dtype=theano.config.floatX))

    # Make sure accumulators are reset.
    features = numpy.array([[2, 1],
                           [1, 3],
                           [1, -1],
                           [2.5, 1]], dtype=theano.config.floatX)

    dataset = IndexableDataset(OrderedDict([('features', features)]))

    data_stream = DataStream(dataset,
                             iteration_scheme=SequentialScheme(num_examples,
                                                               batch_size))
    assert_allclose(DatasetEvaluator([y]).evaluate(data_stream)['y'],
                    numpy.array([7.25, 10], dtype=theano.config.floatX))
    assert_allclose(DatasetEvaluator([z]).evaluate(data_stream)['z'],
                    numpy.array([2], dtype=theano.config.floatX))


def test_concatenate_aggregator():
    num_examples = 4
    batch_size = 2

    features = numpy.array([[2, 3],
                           [2, 9],
                           [2, 4],
                           [5, 1]], dtype=theano.config.floatX)

    dataset = IndexableDataset(OrderedDict([('features', features)]))

    data_stream = DataStream(dataset,
                             iteration_scheme=SequentialScheme(num_examples,
                                                               batch_size))
    x = tensor.matrix('features')
    y = x.sum(axis=0).copy('y')
    z = y.sum(axis=0).copy('z')
    y.tag.aggregation_scheme = Concatenate(y)
    z.tag.aggregation_scheme = Concatenate(z)

    assert_allclose(DatasetEvaluator([y]).evaluate(data_stream)['y'],
                    numpy.array([[4, 12], [7, 5]], dtype=theano.config.floatX))
    assert_allclose(DatasetEvaluator([z]).evaluate(data_stream)['z'],
                    numpy.array([16, 12], dtype=theano.config.floatX))


def test_aggregation_buffer_name_uniqueness():
    x1 = tensor.scalar('x')
    x2 = tensor.scalar('x')
    assert_raises_regex(ValueError, 'unique', AggregationBuffer, [x1, x2])


def test_aggregation_buffer_name_none():
    assert_raises_regex(ValueError, 'must have names',
                        AggregationBuffer, [theano.tensor.scalar()])
