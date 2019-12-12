import numpy as np
import tensorflow as tf
slim = tf.contrib.slim

@slim.add_arg_scope
def lrelu(inputs, leak=0.2, scope="lrelu"):
    """
    https://github.com/tensorflow/tensorflow/issues/4079
    """
    with tf.variable_scope(scope):
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
        return f1 * inputs + f2 * abs(inputs)

def generate_z(batch_size, z_dim):
    return np.random.uniform(-1, 1, size=(batch_size, z_dim)).astype(np.float32)
