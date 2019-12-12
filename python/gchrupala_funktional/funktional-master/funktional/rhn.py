# Based on code in
# https://github.com/julian121266/RecurrentHighwayNetworks/blob/master/theano_rhn.py
# by The Swiss AI lab IDSIA.
import numpy as np
import theano
import theano.tensor as tt
from theano.ifelse import ifelse
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import numbers
import context
from funktional.layer import Layer, WithH0, FixedZeros, Zeros, Identity, Residual, params
from funktional.util import autoassign

floatX = theano.config.floatX

def cast_floatX(n):
  return np.asarray(n, dtype=floatX)

class Linear(Layer):

    def __init__(self, in_size, out_size, bias_init=None, init_scale=0.04):
        autoassign(locals())
        self.w = self.make_param((self.in_size, self.out_size), 'uniform')
        if bias_init is not None:
            self.b = self.make_param((self.out_size,), self.bias_init)

    def make_param(self, shape, init_scheme):
        """Create Theano shared variables, which are used as trainable model parameters."""
        if isinstance(init_scheme, numbers.Number):
            init_value = np.full(shape, init_scheme, floatX)
        elif init_scheme == 'uniform':
            #init_value = self._np_rng.uniform(low=-self.init_scale, high=self.init_scale, size=shape).astype(floatX) # FIXME
            init_value = np.random.uniform(low=-self.init_scale, high=self.init_scale, size=shape).astype(floatX)

        else:
            raise AssertionError('unsupported init_scheme')
        p = theano.shared(init_value)
        return p


    def params(self):
        if self.bias_init is not None:
            return [self.w, self.b]
        else:
            return [self.w]

    def __call__(self, x):
        if self.bias_init is not None:
            return tt.dot(x, self.w) + self.b
        else:
            return tt.dot(x, self.w)

class RHN(Layer):
    """Recurrent Highway Network. Based on
    https://arxiv.org/abs/1607.03474 and
    https://github.com/julian121266/RecurrentHighwayNetworks.

    """
    def __init__(self, size_in, size, recur_depth=1, drop_i=0.75 , drop_s=0.25,
                 init_T_bias=-2.0, init_H_bias='uniform', tied_noise=True, init_scale=0.04, seed=1):
        autoassign(locals())
        self._theano_rng = RandomStreams(self.seed // 2 + 321)
        #self._np_rng = np.random.RandomState(self.seed // 2 + 123)
        # self._is_training = tt.iscalar('is_training')
        hidden_size = self.size
        self.LinearH = Linear(in_size=self.size_in, out_size=hidden_size, bias_init=self.init_H_bias)
        self.LinearT = Linear(in_size=self.size_in, out_size=hidden_size, bias_init=self.init_T_bias)
        self.recurH = []
        self.recurT = []
        for l in range(self.recur_depth):
            if l == 0:
                self.recurH.append(Linear(in_size=hidden_size, out_size=hidden_size))
                self.recurT.append(Linear(in_size=hidden_size, out_size=hidden_size))
            else:
                self.recurH.append(Linear(in_size=hidden_size, out_size=hidden_size, bias_init=self.init_H_bias))
                self.recurT.append(Linear(in_size=hidden_size, out_size=hidden_size, bias_init=self.init_T_bias))


    def apply_dropout(self, x, noise):
        if context.training:
            return noise * x
        else:
            return x

    def get_dropout_noise(self, shape, dropout_p):
        keep_p = 1 - dropout_p
        noise = cast_floatX(1. / keep_p) * self._theano_rng.binomial(size=shape, p=keep_p, n=1, dtype=floatX)
        return noise

    def params(self):
        return params(*[self.LinearH, self.LinearT] + self.recurH + self.recurT)


    def step(self, i_for_H_t, i_for_T_t, h_tm1, noise_s):
        tanh, sigm = tt.tanh, tt.nnet.sigmoid
        noise_s_for_H = noise_s if self.tied_noise else noise_s[0]
        noise_s_for_T = noise_s if self.tied_noise else noise_s[1]

        hidden_size = self.size
        s_lm1 = h_tm1
        for l in range(self.recur_depth):
            s_lm1_for_H = self.apply_dropout(s_lm1, noise_s_for_H)
            s_lm1_for_T = self.apply_dropout(s_lm1, noise_s_for_T)
            if l == 0:
                # On the first micro-timestep of each timestep we already have bias
                # terms summed into i_for_H_t and into i_for_T_t.
                H = tanh(i_for_H_t + self.recurH[l](s_lm1_for_H))
                T = sigm(i_for_T_t + self.recurT[l](s_lm1_for_T))
            else:
                H = tanh(self.recurH[l](s_lm1_for_H))
                T = sigm(self.recurT[l](s_lm1_for_T))
            s_l = (H - s_lm1) * T + s_lm1
            s_lm1 = s_l

        y_t = s_l
        return y_t

    def __call__(self, h0, seq, repeat_h0=1):
        inputs = seq.dimshuffle((1,0,2))
        (_seq_size, batch_size, _) = inputs.shape
        hidden_size = self.size
        # We first compute the linear transformation of the inputs over all timesteps.
        # This is done outside of scan() in order to speed up computation.
        # The result is then fed into scan()'s step function, one timestep at a time.
        noise_i_for_H = self.get_dropout_noise((batch_size, self.size_in), self.drop_i)
        noise_i_for_T = self.get_dropout_noise((batch_size, self.size_in), self.drop_i) if not self.tied_noise else noise_i_for_H

        i_for_H = self.apply_dropout(inputs, noise_i_for_H)
        i_for_T = self.apply_dropout(inputs, noise_i_for_T)

        i_for_H = self.LinearH(i_for_H)
        i_for_T = self.LinearT(i_for_T)

        # Dropout noise for recurrent hidden state.
        noise_s = self.get_dropout_noise((batch_size, hidden_size), self.drop_s)
        if not self.tied_noise:
          noise_s = tt.stack(noise_s, self.get_dropout_noise((batch_size, hidden_size), self.drop_s))

        H0 = tt.repeat(h0, inputs.shape[1], axis=0) if repeat_h0 else h0
        out, _ = theano.scan(self.step,
                             sequences=[i_for_H, i_for_T],
                             outputs_info=[H0],
                             non_sequences = [noise_s])
        return out.dimshuffle((1, 0, 2))


def RHN0(size_in, size, fixed=False, **kwargs):
    """A GRU layer with its own initial state."""
    if fixed:
        return WithH0(FixedZeros(size), RHN(size_in, size, **kwargs))
    else:
        return WithH0(Zeros(size), RHN(size_in, size, **kwargs))


class StackedRHN(Layer):
    """A stack of RHNs.
    """
    def __init__(self, size_in, size, depth=2, residual=False, fixed=False, **kwargs):
#    def __init__(self, size_in, size, depth=2, dropout_prob=0.0, residual=False, fixed=False, **kwargs):
        autoassign(locals())
        f = lambda x: Residual(x) if self.residual else x
        self.layers = [ f(RHN0(self.size, self.size, fixed=self.fixed, **self.kwargs))  for _ in range(1,self.depth) ]
        self.bottom = RHN(self.size_in, self.size, **self.kwargs)
        self.stack = reduce(lambda z, x: x.compose(z), self.layers, Identity())

    def params(self):
        return params(self.bottom, self.stack)

    def __call__(self, h0, inp, repeat_h0=0):
        return self.stack(self.bottom(h0, inp, repeat_h0=repeat_h0))

    def intermediate(self, h0, inp, repeat_h0=0):
        zs = [ self.bottom(h0, inp, repeat_h0=repeat_h0) ]
        for layer in self.layers:
            z = layer(zs[-1])
            zs.append(z)
        return theano.tensor.stack(* zs).dimshuffle((1,2,0,3)) # FIXME deprecated interface

def StackedRHN0(size_in, size, depth, fixed=False, **kwargs):
    """A stacked RHN layer with its own initial state."""
    if fixed:
        return WithH0(FixedZeros(size), StackedRHN(size_in, size, depth, fixed=fixed, **kwargs))
    else:
        return WithH0(Zeros(size), StackedRHN(size_in, size, depth, **kwargs))
