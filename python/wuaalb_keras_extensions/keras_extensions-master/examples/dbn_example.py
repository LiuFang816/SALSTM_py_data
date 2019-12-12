from __future__ import division

import time

import numpy as np
np.random.seed(1234) # seed random number generator
srng_seed = np.random.randint(2**30)

from keras.models import Sequential
from keras.optimizers import SGD

from keras_extensions.logging import log_to_file
from keras_extensions.rbm import GBRBM, RBM
from keras_extensions.dbn import DBN
from keras_extensions.layers import SampleBernoulli
from keras_extensions.initializers import glorot_uniform_sigm

# configuration
input_dim = 100
hidden_dim = 200
batch_size = 10
nb_epoch = 1
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
    dbn = DBN([
        GBRBM(input_dim, 200, init=glorot_uniform_sigm),
        RBM(200, 400, init=glorot_uniform_sigm),
        RBM(400, 300, init=glorot_uniform_sigm),
        RBM(300, 50, init=glorot_uniform_sigm),
        RBM(50, hidden_dim, init=glorot_uniform_sigm)
    ])

    # setup optimizer, loss
    def get_layer_loss(rbm,layer_no):
        return rbm.contrastive_divergence_loss(nb_gibbs_steps=1)
    def get_layer_optimizer(layer_no):
        return SGD((layer_no+1)*lr, 0., decay=0.0, nesterov=False)
    dbn.compile(layer_optimizer=get_layer_optimizer, layer_loss=get_layer_loss)
    
    # do training
    print('Training...')
    begin_time = time.time()

    #callbacks = [momentum_scheduler, rec_err_logger, free_energy_gap_logger]
    dbn.fit(X_train, batch_size, nb_epoch, verbose=1, shuffle=False)

    end_time = time.time()

    print('Training took %f minutes' % ((end_time - begin_time)/60.0))

    # save model parameters
    print('Saving model...')
    dbn.save_weights('example.hdf5', overwrite=True)

    # load model parameters
    print('Loading model...')
    dbn.load_weights('example.hdf5')

    # generate hidden features from input data
    print('Creating inference model...')
    F= dbn.get_forward_inference_layers()
    B= dbn.get_backward_inference_layers()
    inference_model = Sequential()
    for f in F:
        inference_model.add(f)
        inference_model.add(SampleBernoulli(mode='random'))
    for b in B[:-1]:
        inference_model.add(b)
        inference_model.add(SampleBernoulli(mode='random'))
    # last layer is a gaussian layer
    inference_model.add(B[-1])
    
    print('Compiling Theano graph...')
    opt = SGD()
    inference_model.compile(opt, loss='mean_squared_error') # XXX: optimizer and loss are not used!

    print('Doing inference...')
    h = inference_model.predict(dataset)

    print(h)

    print('Done!')

if __name__ == '__main__':
    main()
