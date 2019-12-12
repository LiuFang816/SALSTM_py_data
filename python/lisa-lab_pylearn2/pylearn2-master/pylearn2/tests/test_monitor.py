from __future__ import print_function

import numpy as np
import warnings
from nose.tools import assert_raises
from theano.compat.six.moves import xrange

from theano.compat import exc_message
from theano import shared
from theano import tensor as T

from pylearn2.compat import OrderedDict
from pylearn2.costs.cost import Cost
from pylearn2.datasets.dense_design_matrix import DenseDesignMatrix
from pylearn2.models.model import Model
from pylearn2.models.s3c import S3C, E_Step, Grad_M_Step
from pylearn2.monitor import _err_ambig_data
from pylearn2.monitor import _err_no_data
from pylearn2.monitor import Monitor
from pylearn2.monitor import push_monitor
from pylearn2.space import VectorSpace
from pylearn2.testing.datasets import ArangeDataset
from pylearn2.training_algorithms.default import DefaultTrainingAlgorithm
from pylearn2.utils.iteration import _iteration_schemes, has_uniform_batch_size
from pylearn2.utils import py_integer_types
from pylearn2.utils.serial import from_string
from pylearn2.utils.serial import to_string
from pylearn2.utils import sharedX
from pylearn2.testing.prereqs import ReadVerifyPrereq


class DummyModel(Model):
    def  __init__(self, num_features):
        self.input_space = VectorSpace(num_features)

    def get_default_cost(self):
        return DummyCost()


class DummyDataset(DenseDesignMatrix):
    def __init__(self, num_examples, num_features):
        rng = np.random.RandomState([4, 12, 17])
        super(DummyDataset, self).__init__(
            X=rng.uniform(1., 2., (num_examples, num_features))
        )

    def __getstate__(self):
        raise AssertionError("These unit tests only test Monitor "
                "functionality. If the Monitor tries to serialize a "
                "Dataset, that is an error.")

class DummyCost(Cost):
    def get_data_specs(self, model):
        return (VectorSpace(1), 'dummy')
    
    def expr(self, model, cost_ipt):
        return cost_ipt.sum()

def test_channel_scaling_sequential():
    def channel_scaling_checker(num_examples, mode, num_batches, batch_size):
        num_features = 2
        monitor = Monitor(DummyModel(num_features))
        dataset = DummyDataset(num_examples, num_features)
        monitor.add_dataset(dataset=dataset, mode=mode,
                                num_batches=num_batches, batch_size=batch_size)
        vis_batch = T.matrix()
        mean = vis_batch.mean()
        data_specs = (monitor.model.get_input_space(),
                      monitor.model.get_input_source())
        monitor.add_channel(name='mean', ipt=vis_batch, val=mean, dataset=dataset,
                            data_specs=data_specs)
        monitor()
        assert 'mean' in monitor.channels
        mean = monitor.channels['mean']
        assert len(mean.val_record) == 1
        actual = mean.val_record[0]
        X = dataset.get_design_matrix()
        if batch_size is not None and num_batches is not None:
            total = min(num_examples, num_batches * batch_size)
        else:
            total = num_examples
        expected = X[:total].mean()
        if not np.allclose(expected, actual):
            raise AssertionError("Expected monitor to contain %f but it has "
                                 "%f" % (expected, actual))

    # Specifying num_batches; even split
    yield channel_scaling_checker, 10, 'sequential', 5, None
    # Specifying num_batches; even split
    yield channel_scaling_checker, 10, 'sequential', 2, None
    # Specifying batch_size; even split
    yield channel_scaling_checker, 10, 'sequential', None, 5
    # Specifying batch_size; even split
    yield channel_scaling_checker, 10, 'sequential', None, 2
    # Specifying num_batches; uneven split
    yield channel_scaling_checker, 10, 'sequential', 4, None
    # Specifying num_batches; uneven split
    yield channel_scaling_checker, 10, 'sequential', 3, None
    # Specifying batch_size; uneven split
    yield channel_scaling_checker, 10, 'sequential', None, 3
    # Specifying batch_size; uneven split
    yield channel_scaling_checker, 10, 'sequential', None, 4
    # Specifying both, even split
    yield channel_scaling_checker, 10, 'sequential', 2, 5
    # Specifying both, even split
    yield channel_scaling_checker, 10, 'sequential', 5, 2
    # Specifying both, uneven split, dangling batch
    yield channel_scaling_checker, 10, 'sequential', 3, 4
    # Specifying both, uneven split, non-exhaustive
    yield channel_scaling_checker, 10, 'sequential', 3, 3

def test_counting():
    BATCH_SIZE = 2
    BATCHES = 3
    NUM_FEATURES = 4
    num_examples = BATCHES * BATCH_SIZE
    dataset = DummyDataset( num_examples = num_examples,
            num_features = NUM_FEATURES)
    algorithm = DefaultTrainingAlgorithm( batch_size = BATCH_SIZE,
            batches_per_iter = BATCHES)
    model = S3C( nvis = NUM_FEATURES, nhid = 1,
            irange = .01, init_bias_hid = 0., init_B = 1.,
            min_B = 1., max_B = 1., init_alpha = 1.,
            min_alpha = 1., max_alpha = 1., init_mu = 0.,
            m_step = Grad_M_Step( learning_rate = 0.),
            e_step = E_Step( h_new_coeff_schedule = [ 1. ]))
    algorithm.setup(model = model, dataset = dataset)
    algorithm.train(dataset = dataset)
    if not ( model.monitor.get_batches_seen() == BATCHES):
        raise AssertionError('Should have seen '+str(BATCHES) + \
                ' batches but saw '+str(model.monitor.get_batches_seen()))

    assert model.monitor.get_examples_seen() == num_examples
    assert isinstance(model.monitor.get_examples_seen(), py_integer_types)
    assert isinstance(model.monitor.get_batches_seen(), py_integer_types)

def test_large_examples():
    BATCH_SIZE = 10000
    num_examples = 60002994
    NUM_FEATURES = 3

    model = DummyModel(NUM_FEATURES)
    monitor = Monitor.get_monitor(model)

    monitoring_dataset = DummyDataset(num_examples = num_examples,
            num_features = NUM_FEATURES)

    monitor.add_dataset(monitoring_dataset, 'sequential', batch_size=BATCH_SIZE)

    name = 'z'

    monitor.add_channel(name = name,
            ipt = model.input_space.make_theano_batch(),
            val = 0.,
            data_specs=(model.get_input_space(), model.get_input_source()))

    try:
        monitor()
    except RuntimeError:
        assert False

def test_reject_empty():

    # Test that Monitor raises an error if asked to iterate over 0 batches

    BATCH_SIZE = 2
    num_examples = BATCH_SIZE
    NUM_FEATURES = 3

    model = DummyModel(NUM_FEATURES)
    monitor = Monitor.get_monitor(model)

    monitoring_dataset = DummyDataset(num_examples = num_examples,
            num_features = NUM_FEATURES)

    monitor.add_dataset(monitoring_dataset, 'sequential', batch_size=BATCH_SIZE,
            num_batches = 0)

    name = 'z'

    monitor.add_channel(name = name,
            ipt = model.input_space.make_theano_batch(),
            val = 0.,
            data_specs=(model.get_input_space(), model.get_input_source()))

    try:
        monitor()
    except ValueError:
        return
    assert False

def test_prereqs():

    # Test that prereqs get run before the monitoring channels are computed

    BATCH_SIZE = 2
    num_examples = BATCH_SIZE
    NUM_FEATURES = 3

    model = DummyModel(NUM_FEATURES)
    monitor = Monitor.get_monitor(model)

    monitoring_dataset = DummyDataset(num_examples = num_examples,
            num_features = NUM_FEATURES)

    monitor.add_dataset(monitoring_dataset, 'sequential', batch_size=BATCH_SIZE)

    prereq_counter = sharedX(0.)
    def prereq(*data):
        prereq_counter.set_value(prereq_counter.get_value() + 1.)

    name = 'num_prereq_calls'

    monitor.add_channel(name = name,
            ipt = model.input_space.make_theano_batch(),
            val = prereq_counter,
            prereqs = [ prereq ],
            data_specs=(model.get_input_space(), model.get_input_source()))

    channel = monitor.channels[name]

    assert len(channel.val_record) == 0
    monitor()
    assert channel.val_record == [1]
    monitor()
    assert channel.val_record == [1,2]

def test_revisit():

    # Test that each call to monitor revisits exactly the same data

    BATCH_SIZE = 3
    MAX_BATCH_SIZE = 12
    BATCH_SIZE_STRIDE = 3
    NUM_BATCHES = 10
    num_examples = NUM_BATCHES * BATCH_SIZE

    monitoring_dataset = ArangeDataset(num_examples)

    for mon_batch_size in xrange(BATCH_SIZE, MAX_BATCH_SIZE + 1,
            BATCH_SIZE_STRIDE):
        nums = [1, 3, int(num_examples / mon_batch_size), None]
        
        for mode in sorted(_iteration_schemes):
            if mode == 'even_sequences' and nums is not None:
                # even_sequences iterator does not support specifying a fixed number
                # of minibatches.
                continue
            for num_mon_batches in nums:
                if num_mon_batches is None and mode in ['random_uniform', 'random_slice']:
                    continue

                if has_uniform_batch_size(mode) and \
                   num_mon_batches is not None and \
                   num_mon_batches * mon_batch_size > num_examples:

                    num_mon_batches = int(num_examples / float(mon_batch_size))

                model = DummyModel(1)
                monitor = Monitor.get_monitor(model)

                try:
                    monitor.add_dataset(monitoring_dataset, mode,
                        batch_size=mon_batch_size, num_batches=num_mon_batches)
                except TypeError:
                    monitor.add_dataset(monitoring_dataset, mode,
                        batch_size=mon_batch_size, num_batches=num_mon_batches,
                        seed = 0)

                if has_uniform_batch_size(mode) and num_mon_batches is None:
                    num_mon_batches = int(num_examples / float(mon_batch_size))
                elif num_mon_batches is None:
                    num_mon_batches = int(np.ceil(float(num_examples) /
                                          float(mon_batch_size)))

                batches = [None] * int(num_mon_batches)
                visited = [False] * int(num_mon_batches)

                batch_idx = shared(0)

                class RecorderAndValidator(object):

                    def __init__(self):
                        self.validate = False

                    def __call__(self, *data):
                        """ Initially, records the batches the monitor shows it.
                        When set to validate mode, makes sure the batches shown
                        on the second monitor call match those from the first."""
                        X, = data

                        idx = batch_idx.get_value()
                        batch_idx.set_value(idx + 1)

                        # Note: if the monitor starts supporting variable batch sizes,
                        # take this out. Maybe move it to a new test that the iterator's
                        # uneven property is set accurately
                        warnings.warn("TODO: add unit test that iterators uneven property is set correctly.")
                        # assert X.shape[0] == mon_batch_size

                        if self.validate:
                            previous_batch = batches[idx]
                            assert not visited[idx]
                            visited[idx] = True
                            if not np.allclose(previous_batch, X):
                                print('Visited different data in batch',idx)
                                print(previous_batch)
                                print(X)
                                print('Iteration mode', mode)
                                assert False
                        else:
                            batches[idx] = X
                        # end if
                    # end __call__
                #end class

                prereq = RecorderAndValidator()

                monitor.add_channel(name = 'dummy',
                    ipt = model.input_space.make_theano_batch(),
                    val = 0.,
                    prereqs = [ prereq ],
                    data_specs=(model.get_input_space(),
                                model.get_input_source()))

                try:
                    monitor()
                except RuntimeError:
                    print('monitor raised RuntimeError for iteration mode', mode)
                    raise


                assert None not in batches

                batch_idx.set_value(0)
                prereq.validate = True

                monitor()

                assert all(visited)

def test_prereqs_batch():

    # Test that prereqs get run before each monitoring batch

    BATCH_SIZE = 2
    num_examples = 2 * BATCH_SIZE
    NUM_FEATURES = 3

    model = DummyModel(NUM_FEATURES)
    monitor = Monitor.get_monitor(model)

    monitoring_dataset = DummyDataset(num_examples = num_examples,
            num_features = NUM_FEATURES)

    monitor.add_dataset(monitoring_dataset, 'sequential', batch_size=BATCH_SIZE)

    sign = sharedX(1.)
    def prereq(*data):
        sign.set_value(
                -sign.get_value())

    name = 'batches_should_cancel_to_0'

    monitor.add_channel(name = name,
            ipt = model.input_space.make_theano_batch(),
            val = sign,
            prereqs = [ prereq ],
            data_specs=(model.get_input_space(), model.get_input_source()))

    channel = monitor.channels[name]

    assert len(channel.val_record) == 0
    monitor()
    assert channel.val_record == [0]
    monitor()
    assert channel.val_record == [0,0]


def test_dont_serialize_dataset():

    # Test that serializing the monitor does not serialize the dataset

    BATCH_SIZE = 2
    num_examples = 2 * BATCH_SIZE
    NUM_FEATURES = 3

    model = DummyModel(NUM_FEATURES)
    monitor = Monitor.get_monitor(model)

    monitoring_dataset = DummyDataset(num_examples = num_examples,
            num_features = NUM_FEATURES)
    monitoring_dataset.yaml_src = ""

    monitor.add_dataset(monitoring_dataset, 'sequential', batch_size=BATCH_SIZE)

    monitor()

    to_string(monitor)

def test_serialize_twice():

    # Test that a monitor can be serialized twice
    # with the same result

    model = DummyModel(1)
    monitor = Monitor.get_monitor(model)

    x = to_string(monitor)
    y = to_string(monitor)

    assert x == y

def test_save_load_save():

    """
    Test that a monitor can be saved, then loaded, and then the loaded
    copy can be saved again.
    This only tests that the serialization and deserialization processes
    don't raise an exception. It doesn't test for correctness at all.
    """

    model = DummyModel(1)
    monitor = Monitor.get_monitor(model)

    num_examples = 2
    num_features = 3
    num_batches = 1
    batch_size = 2

    dataset = DummyDataset(num_examples, num_features)
    monitor.add_dataset(dataset=dataset,
                            num_batches=num_batches, batch_size=batch_size)
    vis_batch = T.matrix()
    mean = vis_batch.mean()
    data_specs = (monitor.model.get_input_space(),
                  monitor.model.get_input_source())
    monitor.add_channel(name='mean', ipt=vis_batch, val=mean, dataset=dataset,
                        data_specs=data_specs)

    saved = to_string(monitor)
    monitor = from_string(saved)
    saved_again = to_string(monitor)

def test_valid_after_serialize():

    # Test that serializing the monitor does not ruin it

    BATCH_SIZE = 2
    num_examples = 2 * BATCH_SIZE
    NUM_FEATURES = 3

    model = DummyModel(NUM_FEATURES)
    monitor = Monitor.get_monitor(model)

    monitoring_dataset = DummyDataset(num_examples = num_examples,
            num_features = NUM_FEATURES)
    monitoring_dataset.yaml_src = ""

    monitor.add_dataset(monitoring_dataset, 'sequential', batch_size=BATCH_SIZE)

    to_string(monitor)

    monitor.redo_theano()

def test_deserialize():

    # Test that a monitor can be deserialized

    model = DummyModel(1)
    monitor = Monitor.get_monitor(model)

    x = to_string(monitor)
    monitor = from_string(x)
    y = to_string(monitor)

def test_prereqs_multidataset():

    # Test that prereqs are run on the right datasets

    NUM_DATASETS = 4
    NUM_FEATURES = 3

    model = DummyModel(NUM_FEATURES)
    monitor = Monitor.get_monitor(model)

    prereq_counters = []
    datasets = []
    for i in xrange(NUM_DATASETS):

        batch_size = i + 1
        num_examples = batch_size
        dataset = DummyDataset(num_examples = num_examples,
                num_features = NUM_FEATURES)
        dataset.X[:] = i
        datasets.append(dataset)

        monitor.add_dataset(dataset, 'sequential', batch_size=batch_size)

        prereq_counters.append(sharedX(0.))



    channels = []
    for i in xrange(NUM_DATASETS):
        monitor.add_channel(name = str(i),
                ipt = model.input_space.make_theano_batch(),
                val = prereq_counters[i],
                dataset = datasets[i],
                prereqs = [ ReadVerifyPrereq(i, prereq_counters[i]) ],
                data_specs=(model.get_input_space(), model.get_input_source()))

        channels.append(monitor.channels[str(i)])

    for channel in channels:
        assert len(channel.val_record) == 0
    monitor()
    for channel in channels:
        assert channel.val_record == [1]
    monitor()
    for channel in channels:
        assert channel.val_record == [1,2]

    # check that handling all these datasets did not
    # result in them getting serialized
    to_string(monitor)


def test_reject_bad_add_dataset():

    model = DummyModel(1)
    monitor = Monitor.get_monitor(model)
    dataset = DummyDataset(1,1)

    try:
        monitor.add_dataset([dataset],mode=['sequential', 'shuffled'])
    except ValueError:
        return

    raise AssertionError("Monitor.add_dataset accepted bad arguments to "
            "add_dataset.")

def test_no_data():

    # test that the right error is raised if you
    # add a channel to a monitor that has no datasets

    BATCH_SIZE = 2
    num_examples = BATCH_SIZE
    NUM_FEATURES = 3

    model = DummyModel(NUM_FEATURES)
    monitor = Monitor.get_monitor(model)

    name = 'num_prereq_calls'

    try:
        monitor.add_channel(name = name,
            ipt = model.input_space.make_theano_batch(),
            data_specs = (model.input_space, 'features'),
            val = 0.)
    except ValueError as e:
        assert exc_message(e) == _err_no_data
        return
    assert False

def test_ambig_data():

    # test that the right error is raised if you
    # add a channel to a monitor that has multiple datasets
    # and don't specify the dataset

    BATCH_SIZE = 2
    num_examples = BATCH_SIZE
    NUM_FEATURES = 3

    model = DummyModel(NUM_FEATURES)
    monitor = Monitor.get_monitor(model)

    first = DummyDataset(num_examples = num_examples,
            num_features = NUM_FEATURES)
    second = DummyDataset(num_examples = num_examples,
            num_features = NUM_FEATURES)

    monitor.add_dataset(first, 'sequential', batch_size=BATCH_SIZE)
    monitor.add_dataset(second, 'sequential', batch_size=BATCH_SIZE)


    name = 'num_prereq_calls'

    try:
        monitor.add_channel(name = name,
            ipt = model.input_space.make_theano_batch(),
            val = 0.,
            data_specs=(model.get_input_space(), model.get_input_source()))
    except ValueError as e:
        assert exc_message(e) == _err_ambig_data
        return
    assert False

def test_transfer_experience():

    # Makes sure the transfer_experience flag of push_monitor works

    model = DummyModel(num_features = 3)
    monitor = Monitor.get_monitor(model)
    monitor.report_batch(2)
    monitor.report_batch(3)
    monitor.report_epoch()
    model = push_monitor(model, "old_monitor", transfer_experience=True)
    assert model.old_monitor is monitor
    monitor = model.monitor
    assert monitor.get_epochs_seen() == 1
    assert monitor.get_batches_seen() == 2
    assert monitor.get_epochs_seen() == 1


def test_extra_costs():

    # Makes sure Monitor.setup checks type of extra_costs

    num_features = 3
    model = DummyModel(num_features=num_features)
    dataset = DummyDataset(num_examples=2, num_features=num_features)
    monitor = Monitor.get_monitor(model)
    extra_costs = [model.get_default_cost()]
    assert_raises(AssertionError, monitor.setup, dataset, 
                  model.get_default_cost(), 1, extra_costs=extra_costs)

    extra_costs = OrderedDict()
    extra_costs['Cost'] = model.get_default_cost()
    monitor.setup(dataset, model.get_default_cost(), 1, 
                  extra_costs=extra_costs)


if __name__ == '__main__':
    test_revisit()
