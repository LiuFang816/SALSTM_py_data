import os
import time
import scipy.misc
import numpy as np
import tensorflow as tf
slim = tf.contrib.slim

import ops
import model

class Trainer(object):
    def __init__(self, config):
        filename_queue = tf.train.string_input_producer([config.filename_queue])
        self.global_step = tf.Variable(0, name="global_step")
        self.model = self._build_model(filename_queue, config)
        self.saver = tf.train.Saver()

        if not os.path.exists(config.sampledir):
            os.makedirs(config.sampledir)

        # TODO: histogram summaries of z, D_real, D_fake and G
        self.loss_summaries = tf.summary.merge([
            tf.summary.scalar("loss_D_real", self.model["loss_D_real"]),
            tf.summary.scalar("loss_D_fake", self.model["loss_D_fake"]),
            tf.summary.scalar("loss_D", self.model["loss_D"]),
            tf.summary.scalar("loss_G", self.model["loss_G"])])
        self.summary_writer = tf.summary.FileWriter(config.logdir)

        self.sv = tf.train.Supervisor(
            logdir=config.logdir,
            saver=self.saver,
            summary_op=None,
            summary_writer=self.summary_writer,
            save_model_secs=config.save_model_secs,
            checkpoint_basename=config.checkpoint_basename,
            global_step=self.global_step)

        sess_config = tf.ConfigProto(
            allow_soft_placement=True,
            gpu_options=tf.GPUOptions(allow_growth=True))
        self.sess = self.sv.prepare_or_wait_for_session(config=sess_config)

        self.config = config

    def _build_model(self, filename_queue, config):
        x_ = self._im_from_tfrecords(filename_queue, config)
        # TODO: supervised image generation?
        y_ = tf.placeholder(tf.float32, [None, config.y_dim], name="y")
        z_ = tf.placeholder(tf.float32, [None, config.z_dim], name="z")
        is_training_ = tf.placeholder(tf.bool, name="is_training")

        G, G_end_pts = model.generator(z_, is_training_)
        D_real, D_real_end_pts = model.discriminator(x_, is_training_)
        D_fake, D_fake_end_pts = model.discriminator(G, is_training_, reuse=True)

        with tf.variable_scope("Loss_D"):
            loss_D_real = tf.losses.sigmoid_cross_entropy(
                multi_class_labels=tf.ones_like(D_real), logits=D_real)
            loss_D_fake = tf.losses.sigmoid_cross_entropy(
                multi_class_labels=tf.zeros_like(D_fake), logits=D_fake)
            loss_D = loss_D_real + loss_D_fake

        with tf.variable_scope("Loss_G"):
            loss_G = tf.losses.sigmoid_cross_entropy(
                multi_class_labels=tf.ones_like(D_fake), logits=D_fake)

        with tf.variable_scope("Optimizer_D"):
            vars_D = [var for var in tf.trainable_variables() \
                      if "discriminator" in var.name]
            opt_D = tf.train.AdamOptimizer(config.lr,
                beta1=config.beta1).minimize(loss_D,
                                             self.global_step,
                                             var_list=vars_D)

        with tf.variable_scope("Optimizer_G"):
            vars_G = [var for var in tf.trainable_variables() \
                      if "generator" in var.name]
            opt_G = tf.train.AdamOptimizer(config.lr,
                beta1=config.beta1).minimize(loss_G, var_list=vars_G)

        return {"x": x_, "y": y_, "z": z_, "is_training": is_training_,
                "G": G, "D_real": D_real, "D_fake": D_fake,
                "G_end_pts": G_end_pts,
                "D_real_end_pts": D_real_end_pts,
                "D_fake_end_pts": D_fake_end_pts,
                "loss_D_real": loss_D_real, "loss_D_fake": loss_D_fake,
                "loss_D": loss_D, "loss_G": loss_G,
                "opt_D": opt_D, "opt_G": opt_G}

    def fit(self):
        config = self.config
        for step in range(config.max_steps):
            t1 = time.time()
            z = ops.generate_z(config.batch_size, config.z_dim)

            # train discriminator
            self.sess.run(self.model["opt_D"],
                feed_dict={self.model["z"]: z, self.model["is_training"]: True})

            # train generator
            self.sess.run(self.model["opt_G"],
                feed_dict={self.model["z"]: z, self.model["is_training"]: True})

            # train generator again
            # followed by https://github.com/carpedm20/DCGAN-tensorflow/
            self.sess.run(self.model["opt_G"],
                feed_dict={self.model["z"]: z, self.model["is_training"]: True})
            t2 = time.time()

            if (step+1) % config.summary_every_n_steps == 0:
                summary_feed_dict = {
                   self.model["z"]: z, self.model["is_training"]: False
                }
                self.make_summary(summary_feed_dict, step+1)

            if (step+1) % config.sample_every_n_steps == 0:
                eta = (t2-t1)*(config.max_steps-step+1)
                print("Finished {}/{} step, ETA:{:.2f}s"
                      .format(step+1, config.max_steps, eta), end="\r")

                _, gen = self.sample(10)
                for i in range(10):
                    imname = os.path.join(config.sampledir, 
                                          str(step+1)+"_"+str(i+1)+".jpg")
                    scipy.misc.imsave(imname, gen[i])

    def sample(self, sample_size):
        config = self.config
        z = ops.generate_z(sample_size, config.z_dim)

        return z, self.sample_with_given_z(z)

    def sample_with_given_z(self, z):
        gen = self.sess.run(self.model["G"],
            feed_dict={self.model["z"]: z, self.model["is_training"]: False})

        return (gen+1) / 2.0

    def make_summary(self, feed_dict, step):
        summary = self.sess.run(self.loss_summaries, feed_dict=feed_dict)
        self.sv.summary_computed(self.sess, summary, step)

    # TODO: maybe have to handle shuffle args?
    def _im_from_tfrecords(self, filename_queue, config, shuffle=True):
        capacity = config.min_after_dequeue + 3 * config.batch_size

        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(filename_queue)

        features = tf.parse_single_example(
            serialized_example,
            features={
                "height": tf.FixedLenFeature([], tf.int64),
                "width": tf.FixedLenFeature([], tf.int64),
                "image": tf.FixedLenFeature([], tf.string)
            }
        )

        image = tf.decode_raw(features["image"], tf.uint8)
        height = tf.cast(features["height"], tf.int32)
        width = tf.cast(features["width"], tf.int32)

        image = tf.reshape(image, [height, width, 3])
        resized_image = tf.image.resize_images(images=image, size=[64, 64])
        resized_image = resized_image / 127.5 - 1.0

        images = tf.train.shuffle_batch(
            [resized_image],
            batch_size=config.batch_size,
            capacity=capacity,
            num_threads=config.num_threads,
            min_after_dequeue=config.min_after_dequeue,
            allow_smaller_final_batch=True,
            name="images")

        return tf.cast(images, tf.float32)
