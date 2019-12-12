#coding=utf-8

import tensorflow as tf
import chinaCh.input_data_seg as input_data
import time

def weight_variable ( shape ) :
    initial = tf.truncated_normal( shape , stddev =0.05)
    return tf.Variable ( initial  )

def bias_variable ( shape ) :
    initial = tf.constant(0.01, shape = shape )
    return tf.Variable ( initial )

def max_pool_2x2(x) :
    return tf.nn.max_pool(x , ksize =[1 , 3 , 3 , 1] , strides =[1 , 2 , 2 , 1] , padding = 'SAME')

mnist = input_data.read_data_sets("chinaCh/10ClassComplexBackground", one_hot=True)

x_ = tf.placeholder("float", [None, None ,None ,3])
 
h_conv1_1 = tf.contrib.layers.convolution2d(x_, 100, [3,3])
h_conv1_2 = tf.contrib.layers.convolution2d(h_conv1_1, 100, [3,3])
h_pool1 = max_pool_2x2(h_conv1_2)

h_conv2_1 = tf.contrib.layers.convolution2d(h_pool1, 100, [3,3])
h_conv2_2 = tf.contrib.layers.convolution2d(h_conv2_1, 100, [3,3])
h_pool2 = max_pool_2x2(h_conv2_2)

h_conv3_1 = tf.contrib.layers.convolution2d(h_pool2, 100, [3,3])
h_conv3_2 = tf.contrib.layers.convolution2d(h_conv3_1, 100, [3,3])
#h_pool3 = max_pool_2x2(h_conv3_2)

#h_conv4_1 = tf.contrib.layers.convolution2d(h_pool3, 100, [3,3])
#h_conv4_2 = tf.contrib.layers.convolution2d(h_conv4_1, 100, [3,3])
#h_pool4 = max_pool_2x2(h_conv4_2)

outShape1 = tf.shape(h_pool1)
wt1 = weight_variable([3, 3, 100, 100])
bt1 = bias_variable([100])
deconv1 = tf.nn.relu(tf.nn.conv2d_transpose(h_conv3_2, wt1, outShape1, [1, 2, 2, 1], padding='SAME') + bt1)

outShape2 = tf.shape(h_pool1)
wt2 = weight_variable([3, 3, 100, 100])
bt2 = bias_variable([100])
deconv2 = tf.nn.relu(tf.nn.conv2d_transpose(deconv1, wt2, outShape2, [1, 1, 1, 1], padding='SAME') + bt2)  

outShape3 = tf.pack([tf.shape(x_)[0], tf.shape(x_)[1], tf.shape(x_)[2], 100])
wt3 = weight_variable([3, 3, 100, 100])
bt3 = bias_variable([100])
deconv3 = tf.nn.tanh(tf.nn.conv2d_transpose(deconv2, wt3, outShape3, [1, 2, 2, 1], padding='SAME') + bt3)

outShape4 = tf.pack([tf.shape(x_)[0], tf.shape(x_)[1], tf.shape(x_)[2], 2])
wt4 = weight_variable([3, 3, 2, 100])
bt4 = bias_variable([2])
deconv4 = tf.nn.tanh(tf.nn.conv2d_transpose(deconv3, wt4, outShape4, [1, 1, 1, 1], padding='SAME') + bt4)

logits = tf.reshape(deconv4, (-1, 2))

yr_ = tf.placeholder("float", [None, None, None, 2], name="yr")
labels = tf.to_float(tf.reshape(yr_, (-1, 2)))

softmax = tf.nn.softmax(logits)
cross_entropy = tf.reduce_mean(-tf.reduce_sum( labels * tf.log(softmax), reduction_indices=[1]))
#cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, labels))

train_step=tf.train.AdamOptimizer(0.0001).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1) )
accuracy=tf.reduce_mean(tf.cast(correct_prediction, "float"))

objectCount = tf.reduce_sum(tf.argmax(logits, 1))
labelObjectCount = tf.reduce_sum(tf.argmax(labels, 1))
allCount = tf.reduce_sum(labels)


with tf.Session() as sess:
    #classifySaver = tf.train.Saver()
    sess.run(tf.initialize_all_variables())
    #classifySaver.restore(sess, "word_reflect.ckpt")
    
    print time.ctime()
    
    for i in range (100000) :
        batch = mnist.train.next_batch(50)
        x_train = batch[0].reshape(batch[0].shape[0], 48, 48, 3)
        y_train = batch[1].reshape(batch[1].shape[0], 48, 48, 2)
        if i%100 == 0:
            train_accuracy, train_objectCount, train_labelObjectCount, train_allCount, train_cross_entropy = sess.run([accuracy, objectCount, labelObjectCount, allCount, cross_entropy], feed_dict={x_:x_train, yr_:y_train})
            print "step %d,  training accuracy %g train_objectCount %g, train_labelObjectCount %g, cross_entropy %g"%(i, train_accuracy, train_objectCount/train_allCount, train_labelObjectCount/train_allCount, train_cross_entropy)
        train_step.run(feed_dict ={x_: x_train, yr_ : y_train})
    
    saver = tf.train.Saver()
    save_path = saver.save(sess, "word_reflect.ckpt")
    print("Model saved in file: ", save_path)
          
    x_test = mnist.test.images.reshape(mnist.test.images.shape[0], 48, 48, 3)
    y_test = mnist.test.labels.reshape(mnist.test.labels.shape[0], 48, 48, 2)
    print "test accuracy %g"%accuracy.eval(feed_dict ={x_:x_test, yr_:y_test})
    
    print time.ctime()
