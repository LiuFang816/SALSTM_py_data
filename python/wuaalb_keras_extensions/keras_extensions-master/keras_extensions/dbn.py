import numpy as npy
npy.random.seed(1234) # seed random number generator  
srng_seed = npy.random.randint(2**30) 


from keras_extensions.rbm import GBRBM,RBM
from keras_extensions.initializers import glorot_uniform_sigm
from keras_extensions.layers import SampleBernoulli
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from keras.models import Sequential
from keras.optimizers import SGD
from keras.utils.io_utils import HDF5Matrix
from keras_extensions.models import SingleLayerUnsupervised

import keras.backend as K

class DBN(object):
    

    def __init__(self, rbms):
        self.rbms = rbms

        # biases
        self.bs = []
        # weights
        self.Ws  = []

        # building the list of DBN parameters
        self.bs.append(self.rbms[0].bx)
        for rbm in self.rbms:
            self.bs.append(rbm.bh)
            self.Ws.append(rbm.W)

        # DBN params are in bs and Ws
        
        for indx,rbm in enumerate(self.rbms):
            if indx != 0:
                rbm.bx = self.bs[indx]
            rbm.hx = self.bs[indx+1]

            if indx != 0:
                rbm.params = [self.Ws[indx],  self.bs[indx+1]]
            else:
                rbm.params = [self.Ws[indx], self.bs[indx],  self.bs[indx+1]]
            rbm.srng = RandomStreams(seed=srng_seed)
            # hack to make visible biases of each layer except the first non learnable

    

    def compile(self, layer_optimizer, layer_loss):
        self.get_layer_optimizer = layer_optimizer
        self.get_layer_loss      = layer_loss
        # layer_optimizer will be passed layer_no
        # layer_loss will be passed layer_no and corresponding rbm



    def fit(self, X, batch_size=128, nb_epoch=100, verbose=1, shuffle="batch"):
        for i in xrange(len(self.rbms)):
            self.greedy_layerwise_train(layer_no=i,
                X=X,
                batch_size = batch_size,
                nb_epoch = nb_epoch,
                verbose = verbose,
                shuffle = shuffle
            )

    def greedy_layerwise_train(self,layer_no, X, batch_size=128, nb_epoch=100, verbose=1, shuffle="batch"):

        print ">>>",layer_no
        '''
        if layer_no !=0:
            pre_model = Sequential()
            for i,rbm in enumerate(self.rbms[0:layer_no]):
                pre_model.add(rbm.get_h_given_x_layer((i==0)))
                pre_model.add(SampleBernoulli(mode='random'))

            pre_model.compile(SGD(),loss='mean_squared_error')
        '''

        class Inferencer():
            def __init__(self,dbn, data, layer_no):
                self.dbn = dbn
                self.data = data
                self.layer_no = layer_no
                ''' create a forward sampling model '''
                self.pre_model = Sequential()
                for i,rbm in enumerate(self.dbn.rbms[0:self.layer_no]):
                    self.pre_model.add(rbm.get_h_given_x_layer((i==0)))
                    self.pre_model.add(SampleBernoulli(mode='random'))
                
                if self.layer_no > 0:
                    self.pre_model.compile(SGD(),loss='mean_squared_error')


            def __len__(self):
                return self.data.__len__()

            def __getitem__(self,key):
                X = self.data[key]
                if self.layer_no > 0:
                    ''' do the inference '''
                    X = self.pre_model.predict(X)
                return X

            @property
            def shape(self):
                return self.data.shape()
        
        # preparing data
        X = Inferencer(self, X, layer_no)
        
        # preparing model
        model = SingleLayerUnsupervised()
        model.add(self.rbms[layer_no])
        
        loss = self.get_layer_loss(self.rbms[layer_no], layer_no)
        opt  = self.get_layer_optimizer(layer_no)
        
        model.compile(optimizer=opt, loss=loss)
        '''
        for bs in self.bs:
             print K.get_value(bs)[0:10]
        print ""
        #'''
        model.fit(
            X,
            batch_size = batch_size,
            nb_epoch = nb_epoch,
            verbose= verbose,
            shuffle = shuffle
        )
        '''
        for bs in self.bs:
             print K.get_value(bs)[0:10]
        print ""
        #'''


    def save_weights(self, filepath, overwrite=False):
        # Save weights to HDF5
        import h5py
        import os.path
        # if file exists and should not be overwritten
        if not overwrite and os.path.isfile(filepath):
            import sys
            get_input = input
            if sys.version_info[:2] <= (2, 7):
                get_input = raw_input
            overwrite = get_input('[WARNING] %s already exists - overwrite? [y/n]' % (filepath))
            while overwrite not in ['y', 'n']:
                overwrite = get_input('Enter "y" (overwrite) or "n" (cancel).')
            if overwrite == 'n':
                return
            print('[TIP] Next time specify overwrite=True in save_weights!')

        f = h5py.File(filepath, 'w')
        biases = [K.get_value(x) for x in self.bs]
        weights= [K.get_value(w) for w in self.Ws]
        f.attrs['nb_params'] = len(weights)

        for n, param in enumerate(biases):
            param_name = 'param_bias_{}'.format(n)
            param_dset = f.create_dataset(param_name, param.shape, dtype=param.dtype)
            param_dset[:] = param
        
        for n, param in enumerate(weights):
            param_name = 'param_weight_{}'.format(n)
            param_dset = f.create_dataset(param_name, param.shape, dtype=param.dtype)
            param_dset[:] = param
        f.flush()
        f.close()
    
    def load_weights(self, filepath):
        # Loads weights from HDF5 file
        import h5py
        f = h5py.File(filepath)
        weights = [f['param_weight_{}'.format(p)] for p in range(f.attrs['nb_params'])]
        biases  = [f['param_bias_{}'.format(p)] for p in range(f.attrs['nb_params']+1)]
        for model_wieght,saved_weight in zip(self.Ws,weights):
            K.set_value(model_wieght, saved_weight)
        for model_bias,saved_bias in zip(self.bs,biases):
            K.set_value(model_bias, saved_bias)

        f.close()



    def get_forward_inference_layers(self):
        L = []
        for ind in xrange(len(self.rbms)):
            L.append(
                self.rbms[ind].get_h_given_x_layer((ind==0))
            )
        return L

    def get_backward_inference_layers(self):
        L = []
        for ind in reversed(xrange(len(self.rbms))):
            L.append(
                self.rbms[ind].get_x_given_h_layer()
            )
        return L
