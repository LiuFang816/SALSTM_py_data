from __future__ import division

import time

import numpy as np
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

np.random.seed(1234) # seed random number generator
srng_seed = np.random.randint(2**30)

from keras.models import Sequential
from keras.optimizers import SGD

from keras_extensions.logging import log_to_file
from keras_extensions.models import SingleLayerUnsupervised
from keras_extensions.rbm import GBRBM
from keras_extensions.layers import SampleBernoulli
from keras_extensions.callbacks import make_stepped_schedule, MomentumScheduler#, UnsupervisedLoss1Logger, UnsupervisedLoss2Logger
from keras_extensions.initializers import glorot_uniform_sigm
#from keras_extensions.preprocessing import standardize

# configuration
input_dim = 100
hidden_dim = 200
batch_size = 10
nb_epoch = 10
lr = 0.0001  # small learning rate for GB-RBM
momentum_schedule = [(0, 0.5), (5, 0.9)]  # start momentum at 0.5, then 0.9 after 5 epochs

@log_to_file('example.log')
def main():
    # generate dummy dataset
    nframes = 10000
    dataset = np.random.normal(loc=np.zeros(input_dim), scale=np.ones(input_dim), size=(nframes, input_dim))

    # standardize (in this case superfluous)
    #dataset, mean, stddev = standardize(dataset)

    # split into train and test portion
    ntest   = 1000
    X_train = dataset[:-ntest :]     # all but last 1000 samples for training
    X_test  = dataset[-ntest:, :]    # last 1000 samples for testing
    X_trainsub = dataset[:ntest, :]  # subset of training data with same number of samples as testset
    assert X_train.shape[0] >= X_test.shape[0], 'Train set should be at least size of test set!'

    # setup model structure
    print('Creating training model...')
    rbm = GBRBM(input_dim=input_dim, hidden_dim=hidden_dim, init=glorot_uniform_sigm)
    rbm.srng = RandomStreams(seed=srng_seed)
    train_model = SingleLayerUnsupervised()
    train_model.add(rbm)

    # setup optimizer, loss
    momentum_schedule = make_stepped_schedule([(0, 0.5), (5, 0.9)])
    momentum_scheduler = MomentumScheduler(momentum_schedule)

    opt = SGD(lr, 0., decay=0.0, nesterov=False)

    contrastive_divergence = rbm.contrastive_divergence_loss(nb_gibbs_steps=1)

    # compile theano graph
    print('Compiling Theano graph...')
    train_model.compile(optimizer=opt, loss=contrastive_divergence)

    # additional monitors
    #rec_loss = rbm.reconstruction_loss(nb_gibbs_steps=1)
    #rec_err_logger = UnsupervisedLoss1Logger(X_train, loss=rec_loss, label='  - input reconstruction loss', every_n_epochs=1)
    #rec_err_logger.compile()

    #free_energy_gap = rbm.free_energy_gap
    #free_energy_gap_logger = UnsupervisedLoss2Logger(X_trainsub, X_test, loss=free_energy_gap, label='  - free energy gap', every_n_epochs=1)
    #free_energy_gap_logger.compile()

    # do training
    print('Training...')
    begin_time = time.time()

    #callbacks = [momentum_scheduler, rec_err_logger, free_energy_gap_logger]
    callbacks = [momentum_scheduler]
    train_model.fit(X_train, batch_size, nb_epoch, verbose=1, shuffle=False, callbacks=callbacks)

    end_time = time.time()

    print('Training took %f minutes' % ((end_time - begin_time)/60.0))

    # save model parameters
    print('Saving model...')
    rbm.save_weights('example.hdf5', overwrite=True)

    # load model parameters
    print('Loading model...')
    rbm.load_weights('example.hdf5')

    # generate hidden features from input data
    print('Creating inference model...')
    h_given_x = rbm.get_h_given_x_layer()
    inference_model = Sequential([h_given_x, SampleBernoulli(mode='maximum_likelihood')])

    print('Compiling Theano graph...')
    inference_model.compile(opt, loss='mean_squared_error') # XXX: optimizer and loss are not used!

    print('Doing inference...')
    h = inference_model.predict(dataset)

    print(h)

    print('Done!')

if __name__ == '__main__':
    main()
