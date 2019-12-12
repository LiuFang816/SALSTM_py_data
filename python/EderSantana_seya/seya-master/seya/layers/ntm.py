import numpy as np

import theano
import theano.tensor as T
floatX = theano.config.floatX

from keras.layers.recurrent import Recurrent, GRU, LSTM
from keras import backend as K

from seya.utils import rnn_states
tol = 1e-4


def _wta(X):
    M = K.max(X, axis=-1, keepdims=True)
    R = K.switch(K.equal(X, M), X, 0.)
    return R


def _update_controller(self, inp, h_tm1, M):
    """We have to update the inner RNN inside the NTM, this
    is the function to do it. Pretty much copy+pasta from Keras
    """
    x = T.concatenate([inp, M], axis=-1)
    #1 is for gru, 2 is for lstm
    if len(h_tm1) in [1,2]:
        if hasattr(self.rnn,"get_constants"):
            BW,BU = self.rnn.get_constants(x)
            h_tm1 += (BW,BU)
    # update state
    _, h = self.rnn.step(x, h_tm1)

    return h


def _circulant(leng, n_shifts):
    """
    I confess, I'm actually proud of this hack. I hope you enjoy!
    This will generate a tensor with `n_shifts` of rotated versions the
    identity matrix. When this tensor is multiplied by a vector
    the result are `n_shifts` shifted versions of that vector. Since
    everything is done with inner products, everything is differentiable.

    Paramters:
    ----------
    leng: int > 0, number of memory locations
    n_shifts: int > 0, number of allowed shifts (if 1, no shift)

    Returns:
    --------
    shift operation, a tensor with dimensions (n_shifts, leng, leng)
    """
    eye = np.eye(leng)
    shifts = range(n_shifts//2, -n_shifts//2, -1)
    C = np.asarray([np.roll(eye, s, axis=1) for s in shifts])
    return theano.shared(C.astype(theano.config.floatX))


def _renorm(x):
    return x / (x.sum(axis=1, keepdims=True))


def _softmax(x):
    wt = x.flatten(ndim=2)
    w = T.nnet.softmax(wt)
    return w.reshape(x.shape)  # T.clip(s, 0, 1)


def _cosine_distance(M, k):
    dot = (M * k[:, None, :]).sum(axis=-1)
    nM = T.sqrt((M**2).sum(axis=-1))
    nk = T.sqrt((k**2).sum(axis=-1, keepdims=True))
    return dot / (nM * nk)


class NeuralTuringMachine(Recurrent):
    """ Neural Turing Machines

    Non obvious parameter:
    ----------------------
    shift_range: int, number of available shifts, ex. if 3, avilable shifts are
                 (-1, 0, 1)
    n_slots: number of memory locations
    m_length: memory length at each location

    Known issues:
    -------------
    Theano may complain when n_slots == 1.

    """
    def __init__(self, output_dim, n_slots, m_length, shift_range=3,
                 inner_rnn='gru',
                 init='glorot_uniform', inner_init='orthogonal',
                 input_dim=None, input_length=None, **kwargs):
        self.output_dim = output_dim
        self.n_slots = n_slots
        self.m_length = m_length
        self.shift_range = shift_range
        self.init = init
        self.inner_init = inner_init
        self.inner_rnn = inner_rnn

        self.input_dim = input_dim
        self.input_length = input_length
        if self.input_dim:
            kwargs['input_shape'] = (self.input_length, self.input_dim)
        super(NeuralTuringMachine, self).__init__(**kwargs)

    def build(self):
        input_leng, input_dim = self.input_shape[1:]
        self.input = T.tensor3()

        if self.inner_rnn == 'gru':
            self.rnn = GRU(
                activation='relu',
                input_dim=input_dim+self.m_length,
                input_length=input_leng,
                output_dim=self.output_dim, init=self.init,
                inner_init=self.inner_init)
        elif self.inner_rnn == 'lstm':
            self.rnn = LSTM(
                input_dim=input_dim+self.m_length,
                input_length=input_leng,
                output_dim=self.output_dim, init=self.init,
                forget_bias_init='zero',
                inner_init=self.inner_init)
        else:
            raise ValueError('this inner_rnn is not implemented yet.')

        self.rnn.build()

        # initial memory, state, read and write vecotrs
        self.M = theano.shared((.001*np.ones((1,)).astype(floatX)))
        self.init_h = K.zeros((self.output_dim))
        self.init_wr = self.rnn.init((self.n_slots,))
        self.init_ww = self.rnn.init((self.n_slots,))

        # write
        self.W_e = self.rnn.init((self.output_dim, self.m_length))  # erase
        self.b_e = K.zeros((self.m_length))
        self.W_a = self.rnn.init((self.output_dim, self.m_length))  # add
        self.b_a = K.zeros((self.m_length))

        # get_w  parameters for reading operation
        self.W_k_read = self.rnn.init((self.output_dim, self.m_length))
        self.b_k_read = self.rnn.init((self.m_length, ))
        self.W_c_read = self.rnn.init((self.output_dim, 3))  # 3 = beta, g, gamma see eq. 5, 7, 9
        self.b_c_read = K.zeros((3))
        self.W_s_read = self.rnn.init((self.output_dim, self.shift_range))
        self.b_s_read = K.zeros((self.shift_range))  # b_s lol! not intentional

        # get_w  parameters for writing operation
        self.W_k_write = self.rnn.init((self.output_dim, self.m_length))
        self.b_k_write = self.rnn.init((self.m_length, ))
        self.W_c_write = self.rnn.init((self.output_dim, 3))  # 3 = beta, g, gamma see eq. 5, 7, 9
        self.b_c_write = K.zeros((3))
        self.W_s_write = self.rnn.init((self.output_dim, self.shift_range))
        self.b_s_write = K.zeros((self.shift_range))

        self.C = _circulant(self.n_slots, self.shift_range)

        self.trainable_weights = self.rnn.trainable_weights + [
            self.W_e, self.b_e,
            self.W_a, self.b_a,
            self.W_k_read, self.b_k_read,
            self.W_c_read, self.b_c_read,
            self.W_s_read, self.b_s_read,
            self.W_k_write, self.b_k_write,
            self.W_s_write, self.b_s_write,
            self.W_c_write, self.b_c_write,
            self.M,
            self.init_h, self.init_wr, self.init_ww]

        if self.inner_rnn == 'lstm':
            self.init_c = K.zeros((self.output_dim))
            self.trainable_weights = self.trainable_weights + [self.init_c, ]

    def _read(self, w, M):
        return (w[:, :, None]*M).sum(axis=1)

    def _write(self, w, e, a, M):
        Mtilda = M * (1 - w[:, :, None]*e[:, None, :])
        Mout = Mtilda + w[:, :, None]*a[:, None, :]
        return Mout

    def _get_content_w(self, beta, k, M):
        num = beta[:, None] * _cosine_distance(M, k)
        return _softmax(num)

    def _get_location_w(self, g, s, C, gamma, wc, w_tm1):
        wg = g[:, None] * wc + (1-g[:, None])*w_tm1
        Cs = (C[None, :, :, :] * wg[:, None, None, :]).sum(axis=3)
        wtilda = (Cs * s[:, :, None]).sum(axis=1)
        wout = _renorm(wtilda ** gamma[:, None])
        return wout

    def _get_controller_output(self, h, W_k, b_k, W_c, b_c, W_s, b_s):
        k = T.tanh(T.dot(h, W_k) + b_k)  # + 1e-6
        c = T.dot(h, W_c) + b_c
        beta = T.nnet.relu(c[:, 0]) + 1e-4
        g = T.nnet.sigmoid(c[:, 1])
        gamma = T.nnet.relu(c[:, 2]) + 1.0001
        s = T.nnet.softmax(T.dot(h, W_s) + b_s)
        return k, beta, g, gamma, s

    def get_initial_states(self, X):
        batch_size = X.shape[0]
        init_M = self.M.dimshuffle(0, 'x', 'x').repeat(
            batch_size, axis=0).repeat(self.n_slots, axis=1).repeat(
            self.m_length, axis=2)
        init_M = init_M.flatten(ndim=2)

        init_h = self.init_h.dimshuffle(('x', 0)).repeat(batch_size, axis=0)
        init_wr = self.init_wr.dimshuffle(('x', 0)).repeat(batch_size, axis=0)
        init_ww = self.init_ww.dimshuffle(('x', 0)).repeat(batch_size, axis=0)
        if self.inner_rnn == 'lstm':
            init_c = self.init_c.dimshuffle(('x', 0)).repeat(batch_size, axis=0)
            return [init_M, T.nnet.softmax(init_wr), T.nnet.softmax(init_ww),
                    init_h, init_c]
        else:
            return [init_M, T.nnet.softmax(init_wr), T.nnet.softmax(init_ww),
                    init_h]

    @property
    def output_shape(self):
        input_shape = self.input_shape
        if self.return_sequences:
            return input_shape[0], input_shape[1], self.output_dim
        else:
            return input_shape[0], self.output_dim

    def get_full_output(self, train=False):
        """
        This method is for research and visualization purposes. Use it as
        X = model.get_input()  # full model
        Y = ntm.get_output()    # this layer
        F = theano.function([X], Y, allow_input_downcast=True)
        [memory, read_address, write_address, rnn_state] = F(x)

        if inner_rnn == "lstm" use it as
        [memory, read_address, write_address, rnn_cell, rnn_state] = F(x)

        """
        # input shape: (nb_samples, time (padded with zeros), input_dim)
        X = self.get_input(train)
        assert K.ndim(X) == 3
        if K._BACKEND == 'tensorflow':
            if not self.input_shape[1]:
                raise Exception('When using TensorFlow, you should define ' +
                                'explicitely the number of timesteps of ' +
                                'your sequences. Make sure the first layer ' +
                                'has a "batch_input_shape" argument ' +
                                'including the samples axis.')

        mask = self.get_output_mask(train)
        if mask:
            # apply mask
            X *= K.cast(K.expand_dims(mask), X.dtype)
            masking = True
        else:
            masking = False

        if self.stateful:
            initial_states = self.states
        else:
            initial_states = self.get_initial_states(X)

        states = rnn_states(self.step, X, initial_states,
                            go_backwards=self.go_backwards,
                            masking=masking)
        return states

    def step(self, x, states):
        M_tm1, wr_tm1, ww_tm1 = states[:3]
        # reshape
        M_tm1 = M_tm1.reshape((x.shape[0], self.n_slots, self.m_length))
        # read
        h_tm1 = states[3:]
        k_read, beta_read, g_read, gamma_read, s_read = self._get_controller_output(
            h_tm1[0], self.W_k_read, self.b_k_read, self.W_c_read, self.b_c_read,
            self.W_s_read, self.b_s_read)
        wc_read = self._get_content_w(beta_read, k_read, M_tm1)
        wr_t = self._get_location_w(g_read, s_read, self.C, gamma_read,
                                    wc_read, wr_tm1)
        M_read = self._read(wr_t, M_tm1)

        # update controller
        h_t = _update_controller(self, x, h_tm1, M_read)

        # write
        k_write, beta_write, g_write, gamma_write, s_write = self._get_controller_output(
            h_t[0], self.W_k_write, self.b_k_write, self.W_c_write,
            self.b_c_write, self.W_s_write, self.b_s_write)
        wc_write = self._get_content_w(beta_write, k_write, M_tm1)
        ww_t = self._get_location_w(g_write, s_write, self.C, gamma_write,
                                    wc_write, ww_tm1)
        e = T.nnet.sigmoid(T.dot(h_t[0], self.W_e) + self.b_e)
        a = T.tanh(T.dot(h_t[0], self.W_a) + self.b_a)
        M_t = self._write(ww_t, e, a, M_tm1)

        M_t = M_t.flatten(ndim=2)

        return h_t[0], [M_t, wr_t, ww_t] + h_t
