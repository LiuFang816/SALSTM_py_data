"""Restricted Boltzmann Machine TensorFlow implementation."""

from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from tqdm import tqdm

from yadlt.core import Layers, Loss
from yadlt.core import UnsupervisedModel
from yadlt.utils import tf_utils, utilities


class RBM(UnsupervisedModel):
    """Restricted Boltzmann Machine implementation using TensorFlow.

    The interface of the class is sklearn-like.
    """

    def __init__(
        self, num_hidden, visible_unit_type='bin',
        name='rbm', loss_func='mse', learning_rate=0.01,
        regcoef=5e-4, regtype='none', gibbs_sampling_steps=1,
            batch_size=10, num_epochs=10, stddev=0.1):
        """Constructor.

        :param num_hidden: number of hidden units
        :param loss_function: type of loss function
        :param visible_unit_type: type of the visible units (bin or gauss)
        :param gibbs_sampling_steps: optional, default 1
        :param stddev: default 0.1. Ignored if visible_unit_type is not 'gauss'
        """
        UnsupervisedModel.__init__(self, name)

        self.loss_func = loss_func
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.regtype = regtype
        self.regcoef = regcoef

        self.loss = Loss(self.loss_func)

        self.num_hidden = num_hidden
        self.visible_unit_type = visible_unit_type
        self.gibbs_sampling_steps = gibbs_sampling_steps
        self.stddev = stddev

        self.W = None
        self.bh_ = None
        self.bv_ = None

        self.w_upd8 = None
        self.bh_upd8 = None
        self.bv_upd8 = None

        self.cost = None

        self.input_data = None
        self.hrand = None
        self.vrand = None

    def _train_model(self, train_set, validation_set,
                     train_ref=None, Validation_ref=None):
        """Train the model.

        :param train_set: training set
        :param validation_set: validation set. optional, default None
        :return: self
        """
        pbar = tqdm(range(self.num_epochs))
        for i in pbar:
            self._run_train_step(train_set)

            if validation_set is not None:
                feed = self._create_feed_dict(validation_set)
                err = tf_utils.run_summaries(
                    self.tf_session, self.tf_merged_summaries,
                    self.tf_summary_writer, i, feed, self.cost)
                pbar.set_description("Reconstruction loss: %s" % (err))

    def _run_train_step(self, train_set):
        """Run a training step.

        A training step is made by randomly shuffling the training set,
        divide into batches and run the variable update nodes for each batch.
        :param train_set: training set
        :return: self
        """
        np.random.shuffle(train_set)

        batches = [_ for _ in utilities.gen_batches(train_set,
                                                    self.batch_size)]
        updates = [self.w_upd8, self.bh_upd8, self.bv_upd8]

        for batch in batches:
            self.tf_session.run(updates,
                                feed_dict=self._create_feed_dict(batch))

    def _create_feed_dict(self, data):
        """Create the dictionary of data to feed to tf session during training.

        :param data: training/validation set batch
        :return: dictionary(self.input_data: data, self.hrand: random_uniform,
                            self.vrand: random_uniform)
        """
        return {
            self.input_data: data,
            self.hrand: np.random.rand(data.shape[0], self.num_hidden),
            self.vrand: np.random.rand(data.shape[0], data.shape[1])
        }

    def build_model(self, n_features, regtype='none'):
        """Build the Restricted Boltzmann Machine model in TensorFlow.

        :param n_features: number of features
        :param regtype: regularization type
        :return: self
        """
        self._create_placeholders(n_features)
        self._create_variables(n_features)
        self.encode = self.sample_hidden_from_visible(self.input_data)[0]
        self.reconstruction = self.sample_visible_from_hidden(
            self.encode, n_features)

        hprob0, hstate0, vprob, hprob1, hstate1 = self.gibbs_sampling_step(
            self.input_data, n_features)
        positive = self.compute_positive_association(self.input_data,
                                                     hprob0, hstate0)

        nn_input = vprob

        for step in range(self.gibbs_sampling_steps - 1):
            hprob, hstate, vprob, hprob1, hstate1 = self.gibbs_sampling_step(
                nn_input, n_features)
            nn_input = vprob

        negative = tf.matmul(tf.transpose(vprob), hprob1)

        self.w_upd8 = self.W.assign_add(
            self.learning_rate * (positive - negative) / self.batch_size)

        self.bh_upd8 = self.bh_.assign_add(tf.mul(self.learning_rate, tf.reduce_mean(
            tf.sub(hprob0, hprob1), 0)))

        self.bv_upd8 = self.bv_.assign_add(tf.mul(self.learning_rate, tf.reduce_mean(
            tf.sub(self.input_data, vprob), 0)))

        variables = [self.W, self.bh_, self.bv_]
        regterm = Layers.regularization(variables, self.regtype, self.regcoef)

        self.cost = self.loss.compile(vprob, self.input_data, regterm=regterm)

    def _create_placeholders(self, n_features):
        """Create the TensorFlow placeholders for the model.

        :param n_features: number of features
        :return: self
        """
        self.input_data = tf.placeholder(tf.float32, [None, n_features],
                                         name='x-input')
        self.hrand = tf.placeholder(tf.float32, [None, self.num_hidden],
                                    name='hrand')
        self.vrand = tf.placeholder(tf.float32, [None, n_features],
                                    name='vrand')
        # not used in this model, created just to comply with
        # unsupervised_model.py
        self.input_labels = tf.placeholder(tf.float32)
        self.keep_prob = tf.placeholder(tf.float32, name='keep-probs')

    def _create_variables(self, n_features):
        """Create the TensorFlow variables for the model.

        :param n_features: number of features
        :return: self
        """
        self.W = tf.Variable(tf.truncated_normal(
            shape=[n_features, self.num_hidden], stddev=0.1), name='weights')
        self.bh_ = tf.Variable(tf.constant(0.1, shape=[self.num_hidden]),
                               name='hidden-bias')
        self.bv_ = tf.Variable(tf.constant(0.1, shape=[n_features]),
                               name='visible-bias')

    def gibbs_sampling_step(self, visible, n_features):
        """Perform one step of gibbs sampling.

        :param visible: activations of the visible units
        :param n_features: number of features
        :return: tuple(hidden probs, hidden states, visible probs,
                       new hidden probs, new hidden states)
        """
        hprobs, hstates = self.sample_hidden_from_visible(visible)
        vprobs = self.sample_visible_from_hidden(hprobs, n_features)
        hprobs1, hstates1 = self.sample_hidden_from_visible(vprobs)

        return hprobs, hstates, vprobs, hprobs1, hstates1

    def sample_hidden_from_visible(self, visible):
        """Sample the hidden units from the visible units.

        This is the Positive phase of the Contrastive Divergence algorithm.

        :param visible: activations of the visible units
        :return: tuple(hidden probabilities, hidden binary states)
        """
        hprobs = tf.nn.sigmoid(tf.add(tf.matmul(visible, self.W), self.bh_))
        hstates = utilities.sample_prob(hprobs, self.hrand)

        return hprobs, hstates

    def sample_visible_from_hidden(self, hidden, n_features):
        """Sample the visible units from the hidden units.

        This is the Negative phase of the Contrastive Divergence algorithm.
        :param hidden: activations of the hidden units
        :param n_features: number of features
        :return: visible probabilities
        """
        visible_activation = tf.add(
            tf.matmul(hidden, tf.transpose(self.W)),
            self.bv_
        )

        if self.visible_unit_type == 'bin':
            vprobs = tf.nn.sigmoid(visible_activation)

        elif self.visible_unit_type == 'gauss':
            vprobs = tf.truncated_normal(
                (1, n_features), mean=visible_activation, stddev=self.stddev)

        else:
            vprobs = None

        return vprobs

    def compute_positive_association(self, visible,
                                     hidden_probs, hidden_states):
        """Compute positive associations between visible and hidden units.

        :param visible: visible units
        :param hidden_probs: hidden units probabilities
        :param hidden_states: hidden units states
        :return: positive association = dot(visible.T, hidden)
        """
        if self.visible_unit_type == 'bin':
            positive = tf.matmul(tf.transpose(visible), hidden_states)

        elif self.visible_unit_type == 'gauss':
            positive = tf.matmul(tf.transpose(visible), hidden_probs)

        else:
            positive = None

        return positive

    def load_model(self, shape, gibbs_sampling_steps, model_path):
        """Load a trained model from disk.

        The shape of the model (num_visible, num_hidden) and the number
        of gibbs sampling steps must be known in order to restore the model.
        :param shape: tuple(num_visible, num_hidden)
        :param gibbs_sampling_steps:
        :param model_path:
        :return: self
        """
        n_features, self.num_hidden = shape[0], shape[1]
        self.gibbs_sampling_steps = gibbs_sampling_steps

        self.build_model(n_features)

        init_op = tf.initialize_all_variables()
        self.tf_saver = tf.train.Saver()

        with tf.Session() as self.tf_session:

            self.tf_session.run(init_op)
            self.tf_saver.restore(self.tf_session, model_path)

    def get_parameters(self, graph=None):
        """Return the model parameters in the form of numpy arrays.

        :param graph: tf graph object
        :return: model parameters
        """
        g = graph if graph is not None else self.tf_graph

        with g.as_default():
            with tf.Session() as self.tf_session:
                self.tf_saver.restore(self.tf_session, self.model_path)

                return {
                    'W': self.W.eval(),
                    'bh_': self.bh_.eval(),
                    'bv_': self.bv_.eval()
                }
