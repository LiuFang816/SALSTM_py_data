#coding=utf-8

import tensorflow as tf
import chinaCh.input_data_seg as input_data
import time

def weight_variable ( shape ) :
    initial = tf.truncated_normal( shape , stddev =0.1)
    return tf.Variable ( initial )

def bias_variable ( shape ) :
    initial = tf.constant(0.1 , shape = shape )
    return tf.Variable ( initial )

def conv2d(x , W) :
    return tf.nn.conv2d(x , W , strides =[1 , 1 , 1 , 1] , padding = 'SAME')

def max_pool_2x2(x) :
    return tf.nn.max_pool(x , ksize =[1 , 3 , 3 , 1] , strides =[1 , 2 , 2 , 1] , padding = 'SAME')

mnist = input_data.read_data_sets("chinaCh", one_hot=True)

x = tf.placeholder("float", [None, 48*48*3])
y_ = tf.placeholder("float", [None,10])

x_image = tf.reshape(x, [-1 ,48 ,48 ,3])

W_conv1_1 = weight_variable([3, 3, 3, 100])
b_conv1_1 = bias_variable([100])
h_conv1_1 = tf.nn.relu (conv2d(x_image , W_conv1_1) + b_conv1_1)
W_conv1_2 = weight_variable([3, 3, 100, 100])
b_conv1_2 = bias_variable([100])
h_conv1_2 = tf.nn.relu (conv2d(h_conv1_1 , W_conv1_2) + b_conv1_2)
h_pool1 = max_pool_2x2(h_conv1_2)

W_conv2_1 = weight_variable([3, 3, 100, 100])
b_conv2_1 = bias_variable([100])
h_conv2_1 = tf.nn.relu(conv2d(h_pool1 , W_conv2_1) + b_conv2_1)
W_conv2_2 = weight_variable([3, 3, 100, 100])
b_conv2_2 = bias_variable([100])
h_conv2_2 = tf.nn.relu(conv2d(h_conv2_1 , W_conv2_2) + b_conv2_2)
h_pool2 = max_pool_2x2(h_conv2_2)

W_conv3_1 = weight_variable([3, 3, 100, 100])
b_conv3_1 = bias_variable([100])
h_conv3_1 = tf.nn.relu(conv2d(h_pool2 , W_conv3_1) + b_conv3_1)
W_conv3_2 = weight_variable([3, 3, 100, 100])
b_conv3_2 = bias_variable([100])
h_conv3_2 = tf.nn.relu(conv2d(h_conv3_1 , W_conv3_2) + b_conv3_2)
h_pool3 = max_pool_2x2(h_conv3_2)

W_conv4_1 = weight_variable([3, 3, 100, 100])
b_conv4_1 = bias_variable([100])
h_conv4_1 = tf.nn.relu(conv2d(h_pool3 , W_conv4_1) + b_conv4_1)
W_conv4_2 = weight_variable([3, 3, 100, 100])
b_conv4_2 = bias_variable([100])
h_conv4_2 = tf.nn.relu(conv2d(h_conv4_1 , W_conv4_2) + b_conv4_2)
h_pool4 = max_pool_2x2(h_conv4_2)

W_fc1 = weight_variable([3*3*100, 4096])
b_fc1 = bias_variable([4096])
h_pool_flat = tf.reshape(h_pool4, [-1, 3*3*100])
h_fc1 = tf.nn.relu(tf.matmul(h_pool_flat, W_fc1) + b_fc1)

W_fc2 = weight_variable([4096, 4096])
b_fc2 = bias_variable([4096])
h_fc2 = tf.nn.relu(tf.matmul(h_fc1, W_fc2) + b_fc2)

keep_prob = tf.placeholder("float")
h_fc_drop = tf.nn.dropout(h_fc2, keep_prob)

W_fc3 = weight_variable([4096, 10])
b_fc3 = bias_variable([10])

y_conv = tf.matmul(h_fc_drop , W_fc3) + b_fc3

l2_loss = 0.0001 * (tf.nn.l2_loss(W_conv1_1) + tf.nn.l2_loss(W_conv1_2) + tf.nn.l2_loss(W_conv2_1) + tf.nn.l2_loss(W_conv2_2) + tf.nn.l2_loss(W_conv3_1) + tf.nn.l2_loss(W_conv3_2) + tf.nn.l2_loss(W_conv4_1) + tf.nn.l2_loss(W_conv4_2)) 
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_conv, y_)) + l2_loss
train_step=tf.train.AdamOptimizer(0.0001).minimize(loss)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1) )
accuracy=tf.reduce_mean(tf.cast(correct_prediction, "float"))

print time.ctime()

saver = tf.train.Saver()
sess = tf.InteractiveSession()
sess.run(tf.initialize_all_variables())
saver.restore(sess, "word_classify.ckpt")

print "test accuracy %g"%accuracy.eval(feed_dict ={x:mnist.validation.images, y_:mnist.validation.labels, keep_prob:1.0})

print time.ctime()

