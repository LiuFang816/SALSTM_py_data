# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Example / benchmark for building a PTB LSTM model.

Trains the model described in:
(Zaremba, et. al.) Recurrent Neural Network Regularization
http://arxiv.org/abs/1409.2329

There are 3 supported model configurations:
===========================================
| config | epochs | train | valid  | test
===========================================
| small  | 13     | 37.99 | 121.39 | 115.91
| medium | 39     | 48.45 |  86.16 |  82.07
| large  | 55     | 37.87 |  82.62 |  78.29
The exact results may vary depending on the random initialization.

The hyperparameters used in the model:
- init_scale - the initial scale of the weights
- learning_rate - the initial value of the learning rate
- max_grad_norm - the maximum permissible norm of the gradient
- num_layers - the number of LSTM layers
- num_steps - the number of unrolled steps of LSTM
- hidden_size - the number of LSTM units
- max_epoch - the number of epochs trained with the initial learning rate
- max_max_epoch - the total number of epochs for training
- keep_prob - the probability of keeping weights in the dropout layer
- lr_decay - the decay of the learning rate for each epoch after "max_epoch"
- batch_size - the batch size

The data required for this example is in the data/ dir of the
PTB dataset from Tomas Mikolov's webpage:

$ wget http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz
$ tar xvf simple-examples.tgz

To run:

$ python ptb_word_lm.py --data_path=simple-examples/data/

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import os

import numpy as np
import tensorflow as tf

import reader

flags = tf.flags
logging = tf.logging

flags.DEFINE_string(
    "model", "small",
    "A type of model. Possible options are: small, medium, large.")
#flags.DEFINE_string("data_path", '/home/comp/csshshi/data/tensorflow/lstm_data/data', "data_path")
flags.DEFINE_string("data_path", os.environ['HOME']+'/data/tensorflow/lstm_data/data', "data_path")
flags.DEFINE_integer("batchsize", 32, "mini-batch size")
flags.DEFINE_integer("hiddensize", 256, "hidden units number of each lstm layer")
flags.DEFINE_integer("numlayer", 2, "number of lstm layers")
flags.DEFINE_integer("seqlen", 32, "len of sample")
flags.DEFINE_string("device", '0', "select device id")
flags.DEFINE_integer("iters", 1000, "iterations for profiling")
flags.DEFINE_integer("max_max_epoch", 20, "max epochs for training")

FLAGS = flags.FLAGS


class PTBModel(object):
  """The PTB model."""

  def __init__(self, is_training, config):
    self.batch_size = batch_size = config.batch_size
    self.num_steps = num_steps = config.num_steps
    size = config.hidden_size
    vocab_size = config.vocab_size

    self._input_data = tf.placeholder(tf.int32, [batch_size, num_steps])
    self._targets = tf.placeholder(tf.int32, [batch_size, num_steps])

    # Slightly better results can be obtained with forget gate biases
    # initialized to 1 but the hyperparameters of the model would need to be
    # different than reported in the paper.
    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(size, forget_bias=0.0)
    if is_training and config.keep_prob < 1:
      lstm_cell = tf.nn.rnn_cell.DropoutWrapper(
          lstm_cell, output_keep_prob=config.keep_prob)
    cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * config.num_layers)
    
    # print config
    print("batch_size: ", config.batch_size)
    print("num_steps: ", config.num_steps)
    print("hidden_size: ", config.hidden_size)
    print("num_layers: ", config.num_layers)

    self._initial_state = cell.zero_state(batch_size, tf.float32)

    with tf.device("/cpu:0"):
      embedding = tf.get_variable("embedding", [vocab_size, size])
      inputs = tf.nn.embedding_lookup(embedding, self._input_data)

    if is_training and config.keep_prob < 1:
      inputs = tf.nn.dropout(inputs, config.keep_prob)

    # Simplified version of tensorflow.models.rnn.rnn.py's rnn().
    # This builds an unrolled LSTM for tutorial purposes only.
    # In general, use the rnn() or state_saving_rnn() from rnn.py.
    #
    # The alternative version of the code below is:
    #
    # from tensorflow.models.rnn import rnn
    # inputs = [tf.squeeze(input_, [1])
    #           for input_ in tf.split(1, num_steps, inputs)]
    # outputs, state = rnn.rnn(cell, inputs, initial_state=self._initial_state)
    outputs = []
    state = self._initial_state
    with tf.variable_scope("RNN"):
      for time_step in range(num_steps):
        if time_step > 0: tf.get_variable_scope().reuse_variables()
        (cell_output, state) = cell(inputs[:, time_step, :], state)
        outputs.append(cell_output)

    output = tf.reshape(tf.concat(1, outputs), [-1, size])
    softmax_w = tf.get_variable("softmax_w", [size, vocab_size])
    softmax_b = tf.get_variable("softmax_b", [vocab_size])
    logits = tf.matmul(output, softmax_w) + softmax_b
    loss = tf.nn.seq2seq.sequence_loss_by_example(
        [logits],
        [tf.reshape(self._targets, [-1])],
        [tf.ones([batch_size * num_steps])])
    self._cost = cost = tf.reduce_sum(loss) / batch_size
    self._final_state = state

    if not is_training:
      return

    self._lr = tf.Variable(0.0, trainable=False)
    tvars = tf.trainable_variables()
    grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars),
                                      config.max_grad_norm)
    optimizer = tf.train.GradientDescentOptimizer(self.lr)
    self._train_op = optimizer.apply_gradients(zip(grads, tvars))


  def assign_lr(self, session, lr_value):
    session.run(tf.assign(self.lr, lr_value))

  @property
  def input_data(self):
    return self._input_data

  @property
  def targets(self):
    return self._targets

  @property
  def initial_state(self):
    return self._initial_state

  @property
  def cost(self):
    return self._cost

  @property
  def final_state(self):
    return self._final_state

  @property
  def lr(self):
    return self._lr

  @property
  def train_op(self):
    return self._train_op


class SmallConfig(object):
  """Small config."""
  init_scale = 0.1
  learning_rate = 0.1
  max_grad_norm = 5
  num_layers = 2
  num_steps = 20
  hidden_size = 200
  max_epoch = 1  # 4
  max_max_epoch = 20 
  keep_prob = 1.0
  lr_decay = 0.5
  batch_size = 20
  vocab_size = 10000
  device = 0
  iters = 1000


class MediumConfig(object):
  """Medium config."""
  init_scale = 0.05
  learning_rate = 1.0
  max_grad_norm = 5
  num_layers = 2
  num_steps = 35
  hidden_size = 650
  max_epoch = 1
  max_max_epoch = 1
  keep_prob = 0.5
  lr_decay = 0.8
  batch_size = 20
  vocab_size = 10000
  device = 0
  iters = 1000


class LargeConfig(object):
  """Large config."""
  init_scale = 0.04
  learning_rate = 1.0
  max_grad_norm = 10
  num_layers = 2
  num_steps = 35
  hidden_size = 1500
  max_epoch = 14
  max_max_epoch = 55
  keep_prob = 0.35
  lr_decay = 1 / 1.15
  batch_size = 20
  vocab_size = 10000
  device = 0
  iters = 1000


class TestConfig(object):
  """Tiny config, for testing."""
  init_scale = 0.1
  learning_rate = 1.0
  max_grad_norm = 1
  num_layers = 1
  num_steps = 2
  hidden_size = 2
  max_epoch = 1
  max_max_epoch = 1
  keep_prob = 1.0
  lr_decay = 0.5
  batch_size = 20
  vocab_size = 10000
  device = 0
  iters = 1000


def run_epoch(session, m, data, eval_op, verbose=False):
  """Runs the model on the given data."""
  epoch_size = ((len(data) // m.batch_size) - 1) // m.num_steps
  start_time = time.time()
  costs = 0.0
  iters = 0
  print('m.initial_state:', m.initial_state)
  state = session.run(m.initial_state) #.eval()
  step = 0
  for step, (x, y) in enumerate(reader.ptb_iterator(data, m.batch_size,
                                                    m.num_steps)):
    cost, state, _ = session.run([m.cost, m.final_state, eval_op],
                                 {m.input_data: x,
                                  m.targets: y,
                                  m.initial_state: state})
    costs += cost
    iters += m.num_steps

    if verbose and step % (epoch_size // 10) == 10:
      print("%.3f perplexity: %.3f speed: %.0f wps" %
            (step * 1.0 / epoch_size, np.exp(costs / iters),
             iters * m.batch_size / (time.time() - start_time)))

  print("Time for one epoch, %d iters: %.4f seconds" %
            (step+1, time.time() - start_time))
  average_batch_time = (time.time() - start_time)/(step+1)
  print("Average time per minibatch in this epoch: %.4f seconds" % average_batch_time)
   
  return np.exp(costs / iters), average_batch_time


def get_config():
  config = None
  if FLAGS.model == "small":
    config = SmallConfig()
  elif FLAGS.model == "medium":
    config = MediumConfig()
  elif FLAGS.model == "large":
    config = LargeConfig()
  elif FLAGS.model == "test":
    config = TestConfig()
  else:
    raise ValueError("Invalid model: %s", FLAGS.model)

  config.batch_size = FLAGS.batchsize
  config.hidden_size = FLAGS.hiddensize
  config.num_layers = FLAGS.numlayer
  config.num_steps = FLAGS.seqlen
  config.device = FLAGS.device
  config.iters = FLAGS.iters
  config.max_max_epoch = FLAGS.max_max_epoch
  return config


def main(_):
  if not FLAGS.data_path:
    raise ValueError("Must set --data_path to PTB data directory")

  raw_data = reader.ptb_raw_data(FLAGS.data_path)
  train_data, valid_data, test_data, _ = raw_data

  config = get_config()
  eval_config = get_config()
  eval_config.batch_size = 1
  eval_config.num_steps = 1

  if config.device == '-1':
    tf_dev = '/cpu:0'
  else:
    tf_dev = '/gpu:' + config.device

  print(tf_dev)
  tconfig = tf.ConfigProto(allow_soft_placement=True)
  if tf_dev.find('cpu') >= 0: # cpu version
    num_threads = os.getenv('OMP_NUM_THREADS', 1)
    tconfig = tf.ConfigProto(allow_soft_placement=True, intra_op_parallelism_threads=int(num_threads))
  with tf.Graph().as_default(), tf.device(tf_dev), tf.Session(config=tconfig) as session:
    initializer = tf.random_uniform_initializer(-config.init_scale,
                                                config.init_scale)
    with tf.variable_scope("model", reuse=None, initializer=initializer):
      m = PTBModel(is_training=True, config=config)
    with tf.variable_scope("model", reuse=True, initializer=initializer):
       #mvalid = PTBModel(is_training=False, config=config)
       mtest = PTBModel(is_training=False, config=eval_config)

    tf.initialize_all_variables().run()

    total_average_batch_time = 0.0

    epochs_info = []
    for i in range(config.max_max_epoch):
      #lr_decay = config.lr_decay ** max(i - config.max_epoch, 0.0)
      #m.assign_lr(session, config.learning_rate * lr_decay)
      m.assign_lr(session, config.learning_rate)

      print("Epoch: %d Learning rate: %.3f" % (i + 1, session.run(m.lr)))
      train_perplexity, average_batch_time = run_epoch(session, m, train_data, m.train_op, verbose=True)
      total_average_batch_time += average_batch_time
      print("Epoch: %d Train Perplexity: %.3f" % (i + 1, train_perplexity))
      if i % 2 == 0:
         epochs_info.append('%d:_:%.3f'%(i, train_perplexity)) 
#      valid_perplexity = run_epoch(session, mvalid, valid_data, tf.no_op())
#      print("Epoch: %d Valid Perplexity: %.3f" % (i + 1, valid_perplexity))

    print("average_batch_time: %.6f" % (total_average_batch_time/int(config.max_max_epoch)))
    print('epoch_info:'+','.join(epochs_info))

    test_perplexity, test_average_batch_time = run_epoch(session, mtest, test_data, tf.no_op())
    print("Test Perplexity: %.3f" % test_perplexity)


if __name__ == "__main__":
  tf.app.run()
