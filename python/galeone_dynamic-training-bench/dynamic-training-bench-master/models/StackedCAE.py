#Copyright (C) 2016 Paolo Galeone <nessuno@nerdz.eu>
#
#This Source Code Form is subject to the terms of the Mozilla Public
#License, v. 2.0. If a copy of the MPL was not distributed with this
#file, you can obtain one at http://mozilla.org/MPL/2.0/.
#Exhibit B is not attached; this software is compatible with the
#licenses expressed under Section 1.12 of the MPL v2.
"""Build a stacked CAE"""

import tensorflow as tf
from . import utils
from .interfaces.Autoencoder import Autoencoder


class StackedCAE(Autoencoder):
    """Build a stacked CAE"""

    def _pad(self, input_x, filter_side):
        """
        pads input_x with the right amount of zeros.
        Args:
            input_x: 4-D tensor, [batch_side, widht, height, depth]
            filter_side: used to dynamically determine the padding amount
        Returns:
            input_x padded
        """
        # calculate the padding amount for each side
        amount = filter_side - 1
        # pad the input on top, bottom, left, right, with amount zeros
        return tf.pad(input_x,
                      [[0, 0], [amount, amount], [amount, amount], [0, 0]])

    def get(self, images, train_phase=False, l2_penalty=0.0):
        """ define the model with its inputs.
        Use this function to define the model in training and when exporting the model
        in the protobuf format.
        Args:
            images: model input
            train_phase: set it to True when defining the model, during train
            l2_penalty: float value, weight decay (l2) penalty
        Returns:
            is_training_: tf.bool placeholder enable/disable training ops at run time
            predictions: the model output
        """
        num_layers = 9
        filter_side = 3
        filters_number = 9
        with tf.variable_scope(self.__class__.__name__):
            input_x = tf.identity(images)
            input_padded = self._pad(input_x, filter_side)
            for layer in range(num_layers):
                with tf.variable_scope("layer_" + str(layer)):
                    with tf.variable_scope("encode"):
                        encoding = utils.conv_layer(
                            input_padded, [
                                filter_side, filter_side,
                                input_padded.get_shape()[3].value,
                                filters_number
                            ],
                            1,
                            'VALID',
                            activation=tf.nn.tanh,
                            wd=l2_penalty)
                        if train_phase:
                            encoding = tf.nn.dropout(encoding, 0.5)

                    with tf.variable_scope("decode"):
                        output_x = utils.conv_layer(
                            encoding, [
                                filter_side, filter_side, filters_number,
                                images.get_shape()[3].value
                            ],
                            1,
                            'VALID',
                            activation=tf.nn.tanh)

                        tf.add_to_collection(utils.LOSSES_COLLECTION,
                                             self._mse(input_x, output_x))
                        input_x = tf.stop_gradient(output_x)
                        input_padded = self._pad(input_x, filter_side)

        # The is_training_ placeholder is not used, but we define and return it
        # in order to respect the expected output cardinality of the get method
        is_training_ = tf.placeholder_with_default(
            False, shape=(), name="is_training_")
        return is_training_, output_x

    def _mse(self, input_x, output_x):
        # 1/2n \sum^{n}_{i=i}{(x_i - x'_i)^2}
        return tf.divide(
            tf.reduce_mean(tf.square(tf.subtract(input_x, output_x))),
            2.,
            name="mse")

    def loss(self, predictions, real_values):
        """Return the loss operation between predictions and real_values.
        Add L2 weight decay term if any.
        Args:
            predictions: predicted values
            real_values: real values
        Returns:
            Loss tensor of type float.
        """
        with tf.variable_scope('loss'):
            #tf.add_to_collection(utils.LOSSES_COLLECTION, self._mse(real_values, predictions))
            # mse + weight_decay per layer
            error = tf.add_n(
                tf.get_collection(utils.LOSSES_COLLECTION), name='total_loss')

        return error
