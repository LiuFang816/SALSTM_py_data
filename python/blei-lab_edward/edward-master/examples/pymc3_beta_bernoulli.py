#!/usr/bin/env python
"""A simple coin flipping example. Inspired by Stan's toy example.

The model is written in PyMC3.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import edward as ed
import numpy as np
import pymc3 as pm
import theano
import tensorflow as tf

from edward.models import PyMC3Model, Beta

x_obs = theano.shared(np.zeros(1))
with pm.Model() as pm_model:
  p = pm.Beta('p', 1, 1, transform=None)
  x = pm.Bernoulli('x', p, observed=x_obs)

ed.set_seed(42)
model = PyMC3Model(pm_model)

data = {x_obs: np.array([0, 1, 0, 0, 0, 0, 0, 0, 0, 1])}

qp_a = tf.nn.softplus(tf.Variable(tf.random_normal([])))
qp_b = tf.nn.softplus(tf.Variable(tf.random_normal([])))
qp = Beta(a=qp_a, b=qp_b)

inference = ed.KLqp({'p': qp}, data, model)
inference.run(n_iter=500)
