#coding=utf-8

import os
import sys
import tensorflow as tf
from PIL import Image, ImageFilter
import time
import numpy
import matplotlib

def imageprepare(image):
    """
    This function returns the pixel values.
    The imput is a png file location.
    """
    im = Image.open(image)
    width = im.size[0]
    height = im.size[1]

    imageData = numpy.array(im)
    print imageData.shape
    return (imageData*1.0/255.0).reshape(1, height, width, 3)

def predictImage(imvalue):
    def weight_variable ( shape ) :
        initial = tf.truncated_normal( shape , stddev =0.05)
        return tf.Variable ( initial  )
    
    def bias_variable ( shape ) :
        initial = tf.constant(0.01, shape = shape )
        return tf.Variable ( initial )
    
    def max_pool_2x2(x) :
        return tf.nn.max_pool(x , ksize =[1 , 3 , 3 , 1] , strides =[1 , 2 , 2 , 1] , padding = 'SAME')
    
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
    #epsilon = tf.constant(value=1e-4)
    #logits = logits + epsilon
    softmax = tf.nn.softmax(logits)
    
    init_op = tf.initialize_all_variables()
    saver = tf.train.Saver()
    
    with tf.Session() as sess:
        sess.run(init_op)
        saver.restore(sess, "word_reflect.ckpt")

        prediction = tf.reshape(tf.argmax(softmax, 1), tf.pack([tf.shape(x_)[1], tf.shape(x_)[2]]))
        return prediction.eval(feed_dict={x_: imvalue}, session=sess)

def main(pictureName):
    """
    Main function.
    """
    inputFile = os.path.split(pictureName)
    inputFileName = inputFile[1]
   

    print time.ctime()
    imvalue = imageprepare(pictureName)
    print time.ctime()

    newImageDatas = predictImage(imvalue)
    data = ((1-newImageDatas)*255).astype(numpy.uint8)
    newImage = Image.fromarray(data)

    newImageName = "./" + inputFile[0] + "/" + os.path.splitext(inputFileName)[0] + "_reflect.bmp"
    newImage.save(newImageName)
    print("end")  # first value in list

if __name__ == "__main__":
    main(sys.argv[1])


