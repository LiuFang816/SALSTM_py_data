from __future__ import division

import numpy as np

from keras import  initializations, regularizers, constraints
from keras import backend as K 
from keras.layers.core import Layer, Dense

from .backend import random_binomial

import theano

class RBM(Layer):
    """
    Bernoulli-Bernoulli Restricted Boltzmann Machine (RBM).
    """

    # keras.core.Layer part (modified from keras.core.Dense)
    # ------------------------------------------------------

    def __init__(self, input_dim, hidden_dim, init='glorot_uniform', weights=None, name=None,
                 W_regularizer=None, bx_regularizer=None, bh_regularizer=None, #activity_regularizer=None,
                 W_constraint=None, bx_constraint=None, bh_constraint=None):

        super(RBM, self).__init__()
        self.init = initializations.get(init)
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.input = K.placeholder(ndim = 2)
        self.W = self.init((self.input_dim, self.hidden_dim))
        self.bx = K.zeros((self.input_dim))
        self.bh = K.zeros((self.hidden_dim))

        self.params = [self.W, self.bx, self.bh]

        self.regularizers = []
        self.W_regularizer = regularizers.get(W_regularizer)
        if self.W_regularizer:
            self.W_regularizer.set_param(self.W)
            self.regularizers.append(self.W_regularizer)

        self.bx_regularizer = regularizers.get(bx_regularizer)
        if self.bx_regularizer:
            self.bx_regularizer.set_param(self.bx)
            self.regularizers.append(self.bx_regularizer)

        self.bh_regularizer = regularizers.get(bh_regularizer)
        if self.bh_regularizer:
            self.bh_regularizer.set_param(self.bh)
            self.regularizers.append(self.bh_regularizer)

        #self.activity_regularizer = regularizers.get(activity_regularizer)
        #if self.activity_regularizer:
        #    self.activity_regularizer.set_layer(self)
        #    self.regularizers.append(self.activity_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.bx_constraint = constraints.get(bx_constraint)
        self.bh_constraint = constraints.get(bh_constraint)
        self.constraints = [self.W_constraint, self.bx_constraint, self.bh_constraint]

        if weights is not None:
            self.set_weights(weights)

        if name is not None:
            self.set_name(name)

    def set_name(self, name):
        self.W.name = '%s_W' % name
        self.bx.name = '%s_bx' % name
        self.bh.name = '%s_bh' % name

    @property
    def nb_input(self):
        return 1

    @property
    def nb_output(self):
        return 0 # RBM has no output, use get_h_given_x_layer(), get_x_given_h_layer() instead

    def get_input(self, train=False):
        return self.input

    def get_output(self, train=False):
        return None # RBM has no output, use get_h_given_x_layer(), get_x_given_h_layer() instead

    def get_config(self):
        return {"name": self.__class__.__name__,
                "input_dim": self.input_dim,
                "hidden_dim": self.hidden_dim,
                "init": self.init.__name__,
                "W_regularizer": self.W_regularizer.get_config() if self.W_regularizer else None,
                "bx_regularizer": self.bx_regularizer.get_config() if self.bx_regularizer else None,
                "bh_regularizer": self.bh_regularizer.get_config() if self.bh_regularizer else None,
                #"activity_regularizer": self.activity_regularizer.get_config() if self.activity_regularizer else None,
                "W_constraint": self.W_constraint.get_config() if self.W_constraint else None,
                "bx_constraint": self.bx_constraint.get_config() if self.bx_constraint else None,
                "bh_constraint": self.bh_constraint.get_config() if self.bh_constraint else None}

    # persistence, copied from keras.models.Sequential
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
        weights = self.get_weights()
        f.attrs['nb_params'] = len(weights)
        for n, param in enumerate(weights):
            param_name = 'param_{}'.format(n)
            param_dset = f.create_dataset(param_name, param.shape, dtype=param.dtype)
            param_dset[:] = param
        f.flush()
        f.close()

    def load_weights(self, filepath):
        # Loads weights from HDF5 file
        import h5py
        f = h5py.File(filepath)
        weights = [f['param_{}'.format(p)] for p in range(f.attrs['nb_params'])]
        self.set_weights(weights)
        f.close()
    
    # -------------
    # RBM internals
    # -------------

    def free_energy(self, x):
        """
        Compute free energy for Bernoulli RBM, given visible units.

        The marginal probability p(x) = sum_h 1/Z exp(-E(x, h)) can be re-arranged to the form 
        p(x) = 1/Z exp(-F(x)), where the free energy F(x) = -sum_j=1^H log(1 + exp(x^T W[:,j] + bh_j)) - bx^T x, 
        in case of the Bernoulli RBM energy function.
        """
        wx_b = K.dot(x, self.W) + self.bh
        hidden_term = K.sum(K.log(1 + K.exp(wx_b)), axis=1)
        vbias_term = K.dot(x, self.bx)
        return -hidden_term - vbias_term

    def sample_h_given_x(self, x):
        """
        Draw sample from p(h|x).

        For Bernoulli RBM the conditional probability distribution can be derived to be 
           p(h_j=1|x) = sigmoid(x^T W[:,j] + bh_j).
        """
        h_pre = K.dot(x, self.W) + self.bh          # pre-sigmoid (used in cross-entropy error calculation for better numerical stability)
        h_sigm = K.sigmoid(h_pre)              # mean of Bernoulli distribution ('p', prob. of variable taking value 1), sometimes called mean-field value
        h_samp = random_binomial(shape=h_sigm.shape, n=1, p=h_sigm)
                                                    # random sample
                                                    #   \hat{h} = 1,      if p(h=1|x) > uniform(0, 1)
                                                    #             0,      otherwise
        # pre and sigm are returned to compute cross-entropy
        return h_samp, h_pre, h_sigm

    def sample_x_given_h(self, h):
        """
        Draw sample from p(x|h).

        For Bernoulli RBM the conditional probability distribution can be derived to be 
           p(x_i=1|h) = sigmoid(W[i,:] h + bx_i).
        """

        x_pre = K.dot(h, self.W.T) + self.bx        # pre-sigmoid (used in cross-entropy error calculation for better numerical stability)
        x_sigm = K.sigmoid(x_pre)              # mean of Bernoulli distribution ('p', prob. of variable taking value 1), sometimes called mean-field value
        x_samp = random_binomial(shape=x_sigm.shape, n=1, p=x_sigm)
                                                    # random sample
                                                    #   \hat{x} = 1,      if p(x=1|h) > uniform(0, 1)
                                                    #             0,      otherwise
        # pre and sigm are returned to compute cross-entropy
        return x_samp, x_pre, x_sigm

    def gibbs_xhx(self, x0):
        """
        Perform one step of Gibbs sampling, starting from visible sample.
          h1 ~ p(h|x0)
          x1 ~ p(x|h1)
        """
        h1, h1_pre, h1_sigm = self.sample_h_given_x(x0)
        x1, x1_pre, x1_sigm = self.sample_x_given_h(h1)
        # pre and sigm are returned to compute cross-entropy
        return x1, x1_pre, x1_sigm

    def mcmc_chain(self, x, nb_gibbs_steps):
        """
        Perform Markov Chain Monte Carlo, run k steps of Gibbs sampling, 
        starting from visible data, return point estimate at end of chain.

           x0 (data) -> h1 -> x1 -> ... -> xk (reconstruction, negative sample)
        """

        xi = x
        for i in xrange(nb_gibbs_steps):
            xi, xi_pre, xi_sigm = self.gibbs_xhx(xi)
        x_rec, x_rec_pre, x_rec_sigm = xi, xi_pre, xi_sigm
        
        x_rec = theano.gradient.disconnected_grad(x_rec)    # avoid back-propagating gradient through the Gibbs sampling
                                                            # this is similar to T.grad(.., consider_constant=[chain_end])
                                                            # however, as grad() is called in keras.optimizers.Optimizer, 
                                                            # we do it here instead to avoid having to change Keras' code

        return x_rec, x_rec_pre, x_rec_sigm

    def contrastive_divergence_loss(self, nb_gibbs_steps=1):
        """
        Compute contrastive divergence loss with k steps of Gibbs sampling (CD-k).

        Result is a Theano expression with the form loss = f(x).
        """
        def loss(x):
            x_rec, _, _ = self.mcmc_chain(x, nb_gibbs_steps)
            cd = K.mean(self.free_energy(x)) - K.mean(self.free_energy(x_rec))
            return cd

        return loss

    def reconstruction_loss(self, nb_gibbs_steps=1):
        """
        Compute binary cross-entropy between the binary input data and the reconstruction generated by the model.

        Result is a Theano expression with the form loss = f(x).

        Useful as a rough indication of training progress (see Hinton2010).
        Summed over feature dimensions, mean over samples.
        """

        def loss(x):
            _, pre, _ = self.mcmc_chain(x, nb_gibbs_steps)

            # NOTE:
            #   when computing log(sigmoid(x)) and log(1 - sigmoid(x)) of cross-entropy, 
            #   if x is very big negative, sigmoid(x) will be 0 and log(0) will be nan or -inf
            #   if x is very big positive, sigmoid(x) will be 1 and log(1-0) will be nan or -inf
            #   Theano automatically rewrites this kind of expression using log(sigmoid(x)) = -softplus(-x), which 
            #   is more stable numerically
            #   however, as the sigmoid() function used in the reconstruction is inside a scan() operation, Theano 
            #   doesn't 'see' it and is not able to perform the change; as a work-around we use pre-sigmoid value 
            #   generated inside the scan() and apply the sigmoid here
            #
            # NOTE:
            #   not sure how important this is; in most cases seems to work fine using just T.nnet.binary_crossentropy() 
            #   for instance; keras.objectives.binary_crossentropy() simply clips the value entering the log(); and 
            #   this is only used for monitoring, not calculating gradient
            cross_entropy_loss = -T.mean(T.sum(x*T.log(T.nnet.sigmoid(pre)) + (1 - x)*T.log(1 - T.nnet.sigmoid(pre)), axis=1))

            return cross_entropy_loss
        return loss

    def free_energy_gap(self, x_train, x_test):
        """
        Computes the free energy gap between train and test set, F(x_test) - F(x_train).

        In order to avoid overfitting, we cannot directly monitor if the probability of held out data is 
        starting to decrease, due to the partition function.
        We can however compute the ratio p(x_train)/p(x_test), because here the partition functions cancel out.
        This ratio should be close to 1, if it is > 1, the model may be overfitting.

        The ratio can be compute as,
           r = p(x_train)/p(x_test) = exp(-F(x_train) + F(x_test)).
        Alternatively, we compute the free energy gap,
           gap = F(x_test) - F(x_train),
        where F(x) indicates the mean free energy of test data and a representative subset of 
        training data respectively.
        The gap should around 0 normally, but when it starts to grow, the model may be overfitting.
        However, even when the gap is growing, the probability of the training data may be growing even faster, 
        so the probability of the test data may still be improving.

        See: Hinton, "A Practical Guide to Training Restricted Boltzmann Machines", UTML TR 2010-003, 2010, section 6.
        """
        return T.mean(self.free_energy(x_train)) - T.mean(self.free_energy(x_test))

    def get_h_given_x_layer(self, as_initial_layer=False):
        """
        Generates a new Dense Layer that computes mean of Bernoulli distribution p(h|x), ie. p(h=1|x).
        """
        if  as_initial_layer:
            layer = Dense(input_dim=self.input_dim, output_dim=self.hidden_dim, activation='sigmoid', weights=[self.W.get_value(), self.bh.get_value()])
        else:
            layer = Dense(output_dim=self.hidden_dim, activation='sigmoid', weights=[self.W.get_value(), self.bh.get_value()])
        return layer

    def get_x_given_h_layer(self, as_initial_layer=False):
        """
        Generates a new Dense Layer that computes mean of Bernoulli distribution p(x|h), ie. p(x=1|h).
        """
        if as_initial_layer:
            layer = Dense(input_dim=self.hidden_dim, output_dim=self.input_dim, activation='sigmoid', weights=[self.W.get_value().T, self.bx.get_value()])
        else:
            layer = Dense(output_dim=self.input_dim, activation='sigmoid', weights=[self.W.get_value().T, self.bx.get_value()])
        return layer





class GBRBM(RBM):
    """
    Gaussian-Bernoulli Restricted Boltzmann Machine (GB-RBM).

    This GB-RBM does not learn variances of Gaussian units, but instead fixes them to 1 and 
    uses noise-free reconstructions. Input data should be pre-processed to have zero mean 
    and unit variance along the feature dimensions.

    See: Hinton, "A Practical Guide to Training Restricted Boltzmann Machines", UTML TR 2010-003, 2010, section 13.2.
    """

    def __init__(self, input_dim, hidden_dim, init='glorot_uniform', weights=None, name=None,
                 W_regularizer=None, bx_regularizer=None, bh_regularizer=None, #activity_regularizer=None,
                 W_constraint=None, bx_constraint=None, bh_constraint=None):

        super(GBRBM, self).__init__(input_dim, hidden_dim, init, weights, name,
        W_regularizer, bx_regularizer, bh_regularizer, #activity_regularizer,
        W_constraint, bx_constraint, bh_constraint)

    # inherited RBM functions same as BB-RBM

    # -------------
    # RBM internals
    # -------------

    def free_energy(self, x):
        wx_b = K.dot(x, self.W) + self.bh
        vbias_term = 0.5*K.sum((x - self.bx)**2, axis=1)
        hidden_term = K.sum(K.log(1 + K.exp(wx_b)), axis=1)
        return -hidden_term + vbias_term
    
    # sample_h_given_x() same as BB-RBM
    def sample_x_given_h(self, h):
        """
        Draw sample from p(x|h).

        For Gaussian-Bernoulli RBM the conditional probability distribution can be derived to be 
           p(x_i|h) = norm(x_i; sigma_i W[i,:] h + bx_i, sigma_i^2).
        """
        x_mean = K.dot(h, self.W.T) + self.bx
        x_samp = x_mean
                # variances of the Gaussian units are not learned, 
                # instead we fix them to 1 in the energy function
                # here, instead of sampling from the Gaussian distributions, 
                # we simply take their means; we'll end up with a noise-free reconstruction
        # here last two returns are dummy variables related to Bernoulli RBM base class (returning e.g. x_samp, None, None doesn't work)
        return x_samp, x_samp, x_samp
   
    # gibbs_xhx() same as BB-RBM
    # mcmc_chain() same as BB-RBM

    def reconstruction_loss(self, nb_gibbs_steps=1):
        """
        Compute mean squared error between input data and the reconstruction generated by the model.

        Result is a Theano expression with the form loss = f(x).

        Useful as a rough indication of training progress (see Hinton2010).
        Mean over samples and feature dimensions.
        """
        def loss(x):
            x_rec, _, _ = self.mcmc_chain(x, nb_gibbs_steps)

            return K.mean(K.sqr(x - x_rec))
        return loss

    # free_energy_gap() same as BB-RBM

    # get_h_given_x_layer() same as BB-RBM
    def get_x_given_h_layer(self, as_initial_layer=False):
        """
        Generates a new Dense Layer that computes mean of Gaussian distribution p(x|h).
        """
        if not as_initial_layer:
            layer = Dense(output_dim=self.input_dim, activation='linear', weights=[self.W.get_value().T, self.bx.get_value()])
        else:
            layer = Dense(input_dim=self.hidden_dim, output_dim=self.input_dim, activation='linear', weights=[self.W.get_value().T, self.bx.get_value()])
        return layer
