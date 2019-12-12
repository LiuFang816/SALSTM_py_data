# encoding: utf-8
# Copyright (c) 2015 Grzegorz Chrupała
# Some code adapted from https://github.com/IndicoDataSolutions/Passage
# Copyright (c) 2015 IndicoDataSolutions

import theano
import theano.tensor as T
from util import *
import context
import numpy
from theano.sandbox.rng_mrg import MRG_RandomStreams

def params(*layers):
    return sum([ layer.params() for layer in layers ], [])

def param_count(ps):
    return sum(reduce(lambda x, z: x*z, p.get_value().shape) for p in ps)

class Layer(object):
    """Neural net layer. Maps (a number of) theano tensors to a theano tensor."""
    def __init__(self):
        pass

    def __call__(self, *inp):
        raise NotImplementedError

    def params(self):
        raise NotImplementedError
    
    def compose(self, l2):
        """Compose itself with another layer."""
        return ComposedLayer(self, l2)

    def borrow_params(self, ps):
        """Overwrite parameters with given values."""
        qs = self.params()
        assert len(qs) == len(ps)
        for (q,p) in zip(qs, ps):
            print "Setting value of {}".format(q)
            q.set_value(p)
        
class Identity(Layer):
    """Return the input unmodified."""
    def __call__(self, inp):
        return inp

    def params(self):
        return []
    
class Residual(Layer):
    """Residualizes a layer."""
    def __init__(self, layer):
        autoassign(locals())

    def __call__(self, inp):
        return inp + self.layer(inp)

    def params(self):
        return self.layer.params()
    
class ComposedLayer(Layer):
    
    def __init__(self, first, second):
        autoassign(locals())

    def params(self):
        return params(self.first, self.second)

    def __call__(self, inp):
        return self.first(self.second(inp))

    def intermediate(self, inp):
        x = self.second(inp)
        z = self.first(x)
        return (x,z)

class Embedding(Layer):
    """Embedding (lookup table) layer."""
    def __init__(self, size_in, size_out):
        autoassign(locals())
        self.E = uniform((self.size_in, self.size_out))

    def __call__(self, inp):
        return self.E[inp]

    def params(self):
        return [self.E]
    
    def unembed(self, inp):
        """Invert the embedding."""
        return T.dot(inp, self.E.T)
        
def theano_one_hot(idx, n):
    z = T.zeros((idx.shape[0], n))
    one_hot = T.set_subtensor(z[T.arange(idx.shape[0]), idx], 1)
    return one_hot        

class OneHot(Layer):
    """One-hot encoding of input."""
    def __init__(self, size_in):
        autoassign(locals())

    def params(self):
        return []

    def __call__(self, inp):
        return theano_one_hot(inp.flatten(), self.size_in).reshape((inp.shape[0], inp.shape[1], self.size_in))
                

class Dense(Layer):
    """Fully connected layer."""
    def __init__(self, size_in, size_out, init=orthogonal):
        autoassign(locals())
        self.w = self.init((self.size_in, self.size_out))
        self.b = shared0s((self.size_out))

    def params(self):
        return [self.w, self.b]
        
    def __call__(self, inp):
        return T.dot(inp, self.w) + self.b


class Dropout(Layer):
    """Randomly set `prob` fraction of input units to zero during training."""
    def __init__(self, prob):
        autoassign(locals())
        self.rstream = MRG_RandomStreams(seed=numpy.random.randint(10e6))

    def params(self):
        return []
    
    def __call__(self, inp):
        if self.prob > 0.0:
            keep = 1.0 - self.prob
            if context.training:
                return inp * self.rstream.binomial(inp.shape, p=keep, dtype=theano.config.floatX) / keep
            else:
                return inp
        else:
            return inp

class Sum(Layer):
    """Componentwise sum of inputs."""
    
    def __init__(self, size):
        autoassign(locals())
        self.id = T.alloc(0.0, 1, self.size)

    def params(self):
        return []

    def step(self, x_t, x_tm1):
        return x_tm1 + x_t

    def __call__(self, seq):
        X = seq.dimshuffle((1,0,2))
        H0 = T.repeat(self.id, X.shape[1], axis=0)
        out, _ = theano.scan(self.step, sequences=[X], outputs_info=[H0])
        return out.dimshuffle((1,0,2)) # return the whole sequence of partial sums 
                                       # to be compatible with recurrent layers
    
class GRU_gate_activations(Layer):
    """Gated Recurrent Unit layer. Takes initial hidden state, and a
       sequence of inputs, and returns the sequence of hidden states,
       and the sequences of gate activations.
    """
    def __init__(self, size_in, size, activation=tanh, gate_activation=steeper_sigmoid,
                 init_in=orthogonal, init_recur=orthogonal,
                 identity=False, backward=False):
        autoassign(locals())
        if self.identity:
            self._init_identity()
        else:
            self._init()

    def _init_identity(self, prob=0.9):
        """Initialize layer as identity function."""
        assert self.size_in == self.size
        lp = logit(prob)
        self.w_z = sharedX(numpy.identity(self.size) * lp)
        self.w_r = self.init_in((self.size_in, self.size))

        self.u_z = sharedX(numpy.identity(self.size) * lp)
        self.u_r = self.init_recur((self.size, self.size))

        self.b_z = shared0s((self.size))
        self.b_r = shared0s((self.size))

        self.w_h = sharedX(numpy.identity(self.size))
        self.u_h = sharedX(numpy.zeros((self.size, self.size)))
        self.b_h = shared0s((self.size))   

    def _init(self):
        self.w_z = self.init_in((self.size_in, self.size))
        self.w_r = self.init_in((self.size_in, self.size))

        self.u_z = self.init_recur((self.size, self.size))
        self.u_r = self.init_recur((self.size, self.size))

        self.b_z = shared0s((self.size))
        self.b_r = shared0s((self.size))

        self.w_h = self.init_in((self.size_in, self.size)) 
        self.u_h = self.init_recur((self.size, self.size))
        self.b_h = shared0s((self.size))   

        
    def params(self):
        return [self.w_z, self.w_r, self.w_h, self.u_z, self.u_r, self.u_h, self.b_z, self.b_r, self.b_h]
        
    def step(self, xz_t, xr_t, xh_t, h_tm1, u_z, u_r, u_h):
        z = self.gate_activation(xz_t + T.dot(h_tm1, u_z))
        r = self.gate_activation(xr_t + T.dot(h_tm1, u_r))
        h_tilda_t = self.activation(xh_t + T.dot(r * h_tm1, u_h))
        h_t = (1 - z) * h_tm1 + z * h_tilda_t
        return h_t, r, z

    def __call__(self, h0, seq, repeat_h0=0):
        X = seq.dimshuffle((1,0,2))
        H0 = T.repeat(h0, X.shape[1], axis=0) if repeat_h0 else h0
        x_z = T.dot(X, self.w_z) + self.b_z
        x_r = T.dot(X, self.w_r) + self.b_r
        x_h = T.dot(X, self.w_h) + self.b_h
        out, _ = theano.scan(self.step, 
            sequences=[x_z, x_r, x_h], 
                             outputs_info=[H0, None, None], 
                             non_sequences=[self.u_z, self.u_r, self.u_h],
                             go_backwards=self.backward
        )
        return (out[0].dimshuffle((1,0,2)), out[1].dimshuffle((1,0,2)), out[2].dimshuffle((1,0,2)))

class GRU(Layer):
    """Gated Recurrent Unit layer. Takes initial hidden state, and a
       sequence of inputs, and returns the sequence of hidden states.
    """
    def __init__(self, size_in, size, **kwargs):
        autoassign(locals())
        self.gru = GRU_gate_activations(self.size_in, self.size, **kwargs)

    def params(self):
        return self.gru.params()
    
    def __call__(self, h0, seq, repeat_h0=1):
        H, _, _ = self.gru(h0, seq, repeat_h0=repeat_h0)
        return H

class BidiGRU(Layer):
    """Bidirectional Gated Recurrent Unit layer. Takes initial hidden state, and a
       sequence of inputs, and returns the sequence of hidden states.
    """
    def __init__(self, size_in, size, activation=tanh, gate_activation=steeper_sigmoid, identity=False, **kwargs):
        autoassign(locals())
        self.gru_f = GRU_gate_activations(self.size_in, self.size, activation=self.activation,
                                          gate_activation=self.gate_activation,
                                          identity=self.identity, backward=False, **kwargs)
        self.gru_b = GRU_gate_activations(self.size_in, self.size, activation=self.activation,
                                          gate_activation=self.gate_activation,
                                          identity=self.identity, backward=True, **kwargs)


    def params(self):
        return params(self.gru_f, self.gru_b)
    
    def __call__(self, h0, seq, repeat_h0=1):
        H_f, _, _ = self.gru_f(h0, seq, repeat_h0=repeat_h0)
        H_b, _, _ = self.gru_b(h0, seq, repeat_h0=repeat_h0)
        return H_f + H_b

    def bidi(self, h0, seq, repeat_h0=1):
        H_f, _, _ = self.gru_f(h0, seq, repeat_h0=repeat_h0)
        H_b, _, _ = self.gru_b(h0, seq, repeat_h0=repeat_h0)
        return (H_f, H_b)

        
class Zeros(Layer):
    """Returns a shared variable vector of specified size initialized with zeros.""" 
    def __init__(self, size):
        autoassign(locals())
        self.zeros = theano.shared(numpy.asarray(numpy.zeros((1,self.size)), dtype=theano.config.floatX))
    
    def params(self):
        return [self.zeros]
    
    def __call__(self):
        return self.zeros

class FixedZeros(Layer):
    """Returns a vector of specified size filled with zeros."""
    def __init__(self, size):
        autoassign(locals())
        self.zeros = T.zeros((1, self.size))

    def params(self):
        return []
    
    def __call__(self):
        return self.zeros

class WithH0(Layer):
    """Returns a new Layer which composes 'h0' and 'layer' such that 'h0()' is the initial state of 'layer'."""
    def __init__(self, h0, layer):
        autoassign(locals())

    def params(self):
        return params(self.h0, self.layer)

    def __call__(self, inp):
        return self.layer(self.h0(), inp, repeat_h0=1)

    def bidi(self, inp):
        return self.layer.bidi(self.h0(), inp, repeat_h0=1)

    def intermediate(self, inp):
        return self.layer.intermediate(self.h0(), inp, repeat_h0=1)

def GRUH0(size_in, size, fixed=False, **kwargs):
    """A GRU layer with its own initial state."""
    if fixed:
        return WithH0(FixedZeros(size), GRU(size_in, size, **kwargs))
    else:
        return WithH0(Zeros(size), GRU(size_in, size, **kwargs))

def BidiGRUH0(size_in, size, **kwargs):
    """A BidiGRU layer with its own initial state."""
    return WithH0(Zeros(size), BidiGRU(size_in, size, **kwargs))

class WithDropout(Layer):
    """Composes given layer with a dropout layer."""
    def __init__(self, layer, prob):
        autoassign(locals())
        self.Dropout = Dropout(prob=prob)

    def params(self):
        return params(self.layer, self.Dropout)

    def __call__(self, *args, **kwargs):
        return self.Dropout(self.layer(*args, **kwargs))


def last(x):
    """Returns the last time step of all sequences in x."""
    return x.dimshuffle((1,0,2))[-1]

def first(x):
    """Returns the first time step of all sequences in x."""
    return x.dimshuffle((1,0,2))[0]

class EncoderDecoderGRU(Layer):
    """A pair of GRUs: the first one encodes the input sequence into a
       state, the second one decodes the state into a sequence of states.
   
    Args:
      inp (tensor3) - input sequence
      out_prev (tensor3) - sequence of output elements at position -1
   
    Returns:
      tensor3 - sequence of states (one for each element of output sequence)
    """
    def __init__(self, size_in, size, size_out, encoder=GRUH0, decoder=GRU):
        self.size_in  = size_in
        self.size     = size
        self.size_out = size_out
        self.Encode   = encoder(size_in=self.size_in, size=self.size)
        self.Decode   = decoder(size_in=self.size_out, size=self.size)

    def params(self):
        return params(self.Encode, self.Decode)

    def __call__(self, inp, out_prev):
        return self.Decode(last(self.Encode(inp)), out_prev)    

                 
class StackedGRU(Layer):
    """A stack of GRUs.
       Dropout layers intervene between adjacent GRU layers.
    """
    def __init__(self, size_in, size, depth=2, dropout_prob=0.0, residual=False, fixed=False, **kwargs):
        autoassign(locals())
        f = lambda x: Residual(x) if self.residual else x
        self.layers = [ f(GRUH0(self.size, self.size, fixed=self.fixed, **self.kwargs)).compose(Dropout(prob=self.dropout_prob))
                            for _ in range(1,self.depth) ]
        self.bottom = GRU(self.size_in, self.size, **self.kwargs)
        self.Dropout0 = Dropout(prob=self.dropout_prob)
        self.stack = reduce(lambda z, x: x.compose(z), self.layers, Identity())

    def params(self):
        return params(self.Dropout0, self.bottom, self.stack)

    def __call__(self, h0, inp, repeat_h0=0):
        return self.stack(self.bottom(h0, self.Dropout0(inp), repeat_h0=repeat_h0))

    def intermediate(self, h0, inp, repeat_h0=0):
        zs = [ self.bottom(h0, self.Dropout0(inp), repeat_h0=repeat_h0) ]
        for layer in self.layers:
            z = layer(zs[-1])
            zs.append(z)
        return theano.tensor.stack(* zs).dimshuffle((1,2,0,3)) # FIXME deprecated interface

    def grow_id(self, identity=True):
        """Add another layer on top, initialized to the identity function."""
        self.stack = GRUH0(self.size, self.size, identity=identity, **self.kwargs).compose(Dropout(prob=self.dropout_prob)).compose(self.stack)

    def grow(self, ps):
        """Add another layer on top, initialized to given parameter values."""
        gruh0 = GRUH0(self.size, self.size, identity=identity, **self.kwargs)
        gruh0.borrow_params(ps)
        self.stack = gruh0.compose(Dropout(prob=self.dropout_prob)).compose(self.stack)
    
def StackedGRUH0(size_in, size, depth, fixed=False, **kwargs):
    """A stacked GRU layer with its own initial state."""
    if fixed:
        return WithH0(FixedZeros(size), StackedGRU(size_in, size, depth, fixed=fixed, **kwargs))
    else:
        return WithH0(Zeros(size), StackedGRU(size_in, size, depth, **kwargs))

class Convolution1D(Layer):
    """A one-dimensional convolutional layer.
    """
    def __init__(self, size_in, length, size, stride=1, activation=linear):
        autoassign(locals())
        self.W_shape = (self.size, self.size_in, self.length, 1)
        self.W = glorot_uniform(self.W_shape)
        
    def __call__(self, seq):
        seq = expand_dims(seq, -1).dimshuffle((0,2,1,3))
        #result = T.nnet.conv2d(seq, self.W, border_mode='full', subsample=(self.stride, 1))
        # T.nnet.conv2d crashes when using non-unit stride
        result = theano.sandbox.cuda.dnn.dnn_conv(seq, self.W, border_mode='full', subsample=(self.stride, 1))
        result = squeeze(result, 3).dimshuffle((0,2,1))
        return self.activation(result)
    
    def params(self):
        return [self.W]

# Functions copied from Keras:
# https://github.com/fchollet/keras/blob/master/keras/backend/theano_backend.py

def expand_dims(x, dim=-1):
    '''Add a 1-sized dimension at index "dim".
    '''
    pattern = [i for i in range(x.type.ndim)]
    if dim < 0:
        if x.type.ndim == 0:
            dim = 0
        else:
            dim = dim % x.type.ndim + 1
    pattern.insert(dim, 'x')
    return x.dimshuffle(pattern)

def squeeze(x, axis):
    '''Remove a 1-dimension from the tensor at index "axis".
    '''
    broadcastable = x.broadcastable[:axis] + x.broadcastable[axis+1:]
    x = T.patternbroadcast(x, [i == axis for i in range(x.type.ndim)])
    x = T.squeeze(x)
    x = T.patternbroadcast(x, broadcastable)
    return x


def softmax_time(x):
    """Input has shape Batch x Time x 1. Return softmax over dimension T."""
    return T.nnet.softmax(x.reshape((x.shape[0]*x.shape[2], x.shape[1]))).reshape(x.shape)
                          
class Attention(Layer):
    """Parameterized weighted average of a sequence of states."""
    def __init__(self, size_in, size=512, activation=tanh):
        autoassign(locals())
        self.Regress1 = Dense(size_in=self.size_in, size_out=self.size)
        self.Regress2 = Dense(size_in=self.size, size_out=1)
        
    def params(self):
        return params(self.Regress1, self.Regress2)
    
    def __call__(self, h):
        alpha = softmax_time(self.Regress2(self.activation(self.Regress1(h))))
        return T.sum(alpha.repeat(self.size_in, axis=2) * h, axis=1)
        
