# Copyright 2015 Google Inc. All Rights Reserved.
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

"""Converts CIFAR10 png to TFRecords file format with Example protos."""

import os

import tensorflow as tf

import input_data



tf.app.flags.DEFINE_string('directory', 'ckpt',
                           'Directory to download ckpt files and write the '
                           'converted result')
tf.app.flags.DEFINE_integer('validation_size', 10000,
                            'Number of examples to separate from the training '
                            'ckpt for the validation set.')
FLAGS = tf.app.flags.FLAGS


def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def convert_to(images, labels, name):
  num_examples = labels.shape[0]
  print('labels shape is ', labels.shape[0])
  if images.shape[0] != num_examples:
    raise ValueError("Images size %d does not match label size %d." %
                     (images.shape[0], num_examples))
  rows = images.shape[1]
  cols = images.shape[2]
  depth = images.shape[3]

  filename = os.path.join(FLAGS.directory, name + '.tfrecords')
  print('Writing', filename)
  writer = tf.python_io.TFRecordWriter(filename)
  for index in range(num_examples):
    image_raw = images[index].tostring()
    example = tf.train.Example(features=tf.train.Features(feature={
        'height': _int64_feature(rows),
        'width': _int64_feature(cols),
        'depth': _int64_feature(depth),
        'label': _int64_feature(int(labels[index])),
        'image_raw': _bytes_feature(image_raw)}))
    writer.write(example.SerializeToString())


def main(argv):

  # Extract it into numpy arrays.
  train_images = input_data.read_images_from()
  train_labels = input_data.read_labels_from()
  # test_images = input_data.extract_images(test_images_filename)
  # test_labels = input_data.extract_labels(test_labels_filename)

  print(train_images.shape)

  # Generate a validation set.
  validation_images = train_images[:FLAGS.validation_size, :, :, :]
  validation_labels = train_labels[:FLAGS.validation_size]
  train_images = train_images[FLAGS.validation_size:, :, :, :]
  train_labels = train_labels[FLAGS.validation_size:]
  # TODO: create test.tfrecords to run tests after training

  # Convert to Examples and write the result to TFRecords.
  convert_to(train_images, train_labels, 'train')
  convert_to(validation_images, validation_labels, 'validation')
  # convert_to(test_images, test_labels, 'test')


if __name__ == '__main__':
  tf.app.run()
