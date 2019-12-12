# -*- coding: utf-8 -*-
import sugartensor as tf
import matplotlib.pyplot as plt

# set log level to debug
tf.sg_verbosity(10)

#
# hyper parameters
#

batch_size = 100
z_dim = 50


#
# create generator
#

# random uniform seed
z = tf.random_uniform((batch_size, z_dim))

with tf.sg_context(name='generator', size=4, stride=2, act='relu', bn=True):
    # generator network
    gen = (z.sg_dense(dim=1024)
           .sg_dense(dim=7*7*128)
           .sg_reshape(shape=(-1, 7, 7, 128))
           .sg_upconv(dim=64)
           .sg_upconv(dim=1, act='sigmoid', bn=False)
           .sg_squeeze())


#
# draw samples
#

with tf.Session() as sess:
    tf.sg_init(sess)
    # restore parameters
    saver = tf.train.Saver()
    saver.restore(sess, tf.train.latest_checkpoint('asset/train/ckpt'))

    # run generator
    imgs = sess.run(gen)

    # plot result
    _, ax = plt.subplots(10, 10, sharex=True, sharey=True)
    for i in range(10):
        for j in range(10):
            ax[i][j].imshow(imgs[i * 10 + j], 'gray')
            ax[i][j].set_axis_off()
    plt.savefig('asset/train/sample.png', dpi=600)
    tf.sg_info('Sample image saved to "asset/train/sample.png"')
    plt.close()
