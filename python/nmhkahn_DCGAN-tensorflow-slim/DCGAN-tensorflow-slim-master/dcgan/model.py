import tensorflow as tf
slim = tf.contrib.slim

import ops

def batch_norm_params(is_training):
    return {
        "decay": 0.9,
        "epsilon": 1e-5,
        "scale": True,
        "updates_collections": None,
        "is_training": is_training
    }


def gen_arg_scope(is_training=True, outputs_collections=None):
    with slim.arg_scope([slim.conv2d_transpose, slim.fully_connected],
        weights_initializer=tf.truncated_normal_initializer(stddev=0.02),
        activation_fn=tf.nn.relu,
        normalizer_fn=slim.batch_norm,
        normalizer_params=batch_norm_params(is_training),
        outputs_collections=outputs_collections):
        with slim.arg_scope([slim.conv2d_transpose],
            kernel_size=[5, 5], stride=2, padding="SAME") as arg_scp:
            return arg_scp


def disc_arg_scope(is_training=True, outputs_collections=None):
    with slim.arg_scope([slim.conv2d],
        weights_initializer=tf.truncated_normal_initializer(stddev=0.02),
        activation_fn=ops.lrelu,
        normalizer_fn=slim.batch_norm,
        normalizer_params=batch_norm_params(is_training),
        kernel_size=[5, 5], stride=2, padding="SAME",
        outputs_collections=outputs_collections) as arg_scp:
        return arg_scp


def generator(z, is_training, y=None, scope=None):
    with tf.variable_scope(scope or "generator") as scp:
        end_pts_collection = scp.name+"end_pts"
        with slim.arg_scope(gen_arg_scope(is_training, end_pts_collection)):
            net = slim.fully_connected(z, 4*4*512,
                                       normalizer_fn=None,
                                       normalizer_params=None,
                                       scope="projection")
            net = tf.reshape(net, [-1, 4, 4, 512])
            net = slim.batch_norm(net, scope="batch_norm",
                                  **batch_norm_params(is_training))
            net = slim.conv2d_transpose(net, 256, scope="conv_tp0")
            net = slim.conv2d_transpose(net, 128, scope="conv_tp1")
            net = slim.conv2d_transpose(net, 64, scope="conv_tp2")
            net = slim.conv2d_transpose(net, 3,
                                        activation_fn=tf.nn.tanh,
                                        normalizer_fn=None,
                                        normalizer_params=None,
                                        scope="conv_tp3")
            end_pts = slim.utils.convert_collection_to_dict(end_pts_collection)
    return net, end_pts


def discriminator(inputs, is_training, y=None, reuse=None, scope=None):
    with tf.variable_scope(scope or "discriminator", values=[inputs], reuse=reuse) as scp:
        end_pts_collection = scp.name+"end_pts"
        with slim.arg_scope(disc_arg_scope(is_training, end_pts_collection)):
            net = slim.conv2d(inputs, 64,
                              normalizer_fn=None,
                              normalizer_params=None,
                              scope="conv0")
            net = slim.conv2d(net, 128, scope="conv1")
            net = slim.conv2d(net, 256, scope="conv2")
            net = slim.conv2d(net, 512, scope="conv3")
            net = slim.conv2d(net, 1,
                              activation_fn=None,
                              kernel_size=[4, 4], stride=1, padding="VALID",
                              normalizer_fn=None,
                              normalizer_params=None,
                              scope="conv4")
            end_pts = slim.utils.convert_collection_to_dict(end_pts_collection)
            net = tf.squeeze(net, [1, 2], name="squeeze")
    return net, end_pts
