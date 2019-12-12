#coding=utf-8

import tensorflow as tf
import chinaCh.input_data_gray_classify as input_data
import time


class_num = 3790

def max_pool_2x2(x) :
    return tf.nn.max_pool(x , ksize =[1 , 3 , 3 , 1] , strides =[1 , 2 , 2 , 1] , padding = 'SAME')

data = input_data.read_data_sets("createchina/3790Class")

x = tf.placeholder("float", [None, 48*48* 1])
y_label = tf.placeholder("int32", [None])

x_image = tf.reshape(x, [-1 ,48 ,48 , 1])

h_conv1_1 = tf.contrib.layers.convolution2d(x_image, 100, [3,3])
h_conv1_2 = tf.contrib.layers.convolution2d(h_conv1_1, 100, [3,3])
h_pool1 = max_pool_2x2(h_conv1_2)

h_conv2_1 = tf.contrib.layers.convolution2d(h_pool1, 100, [3,3])
h_conv2_2 = tf.contrib.layers.convolution2d(h_conv2_1, 100, [3,3])
h_pool2 = max_pool_2x2(h_conv2_2)

h_conv3_1 = tf.contrib.layers.convolution2d(h_pool2, 100, [3,3])
h_conv3_2 = tf.contrib.layers.convolution2d(h_conv3_1, 100, [3,3])
h_pool3 = max_pool_2x2(h_conv3_2)

h_conv4_1 = tf.contrib.layers.convolution2d(h_pool3, 100, [3,3])
h_conv4_2 = tf.contrib.layers.convolution2d(h_conv4_1, 100, [3,3])
h_pool4 = max_pool_2x2(h_conv4_2)

h_pool_flat = tf.reshape(h_pool4, [-1, 3*3*100])
h_fc1 = tf.contrib.layers.fully_connected(h_pool_flat, 4096, activation_fn=tf.nn.relu)
h_fc2 = tf.contrib.layers.fully_connected(h_fc1, 4096, activation_fn=tf.nn.relu)

keep_prob = tf.placeholder("float")
h_fc_drop = tf.nn.dropout(h_fc2, keep_prob)

y_ = tf.contrib.layers.fully_connected(h_fc_drop, class_num, activation_fn=None)

loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(y_, y_label))
train_step=tf.train.AdamOptimizer(0.0001).minimize(loss)
correct_prediction = tf.equal(tf.cast(tf.argmax(y_,1), "int32"), y_label)
accuracy=tf.reduce_mean(tf.cast(correct_prediction, "float"))

print time.ctime()

saver = tf.train.Saver()
sess = tf.InteractiveSession()
sess.run(tf.initialize_all_variables())

for i in range (100000) :
    batch = data.train.next_batch(50)
    if i%100 == 0:
        train_accuracy, train_loss = sess.run([accuracy, loss], feed_dict={x:batch[0], y_label:batch[1], keep_prob:1.0})
        print "step %d,  training accuracy %g, train_loss %g"%(i, train_accuracy, train_loss)
    train_step.run(feed_dict ={x: batch[0], y_label:batch[1], keep_prob: 0.5})
      
print "test accuracy %g"%accuracy.eval(feed_dict ={x:data.test.images, y_label:data.test.labels, keep_prob:1.0})

save_path = saver.save(sess, "word_classify.ckpt")
print("Model saved in file: ", save_path)
print time.ctime()

