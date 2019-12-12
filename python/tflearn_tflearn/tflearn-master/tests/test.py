'''
    This file contains test cases for tflearn
'''

import tensorflow as tf
import tflearn
import unittest

class TestActivations(unittest.TestCase):
    '''
        This class contains test cases for the functions in tflearn/activations.py
    '''
    PLACES = 4 # Number of places to match when testing floating point values

    def test_linear(self):
        f = tflearn.linear

        # Case 1
        x = tf.placeholder(tf.float32, shape=())
        self.assertEqual(f(x), x)

        # Case 2
        x = tf.placeholder(tf.int64, shape=())
        self.assertEqual(f(x), x)

    def test_tanh(self):
        f = tflearn.tanh
        x = tf.placeholder(tf.float32, shape=())
        
        with tf.Session() as sess:
            # Case 1
            self.assertEqual(sess.run(f(x), feed_dict={x:0}), 0)

            # Case 2
            self.assertAlmostEqual(sess.run(f(x), feed_dict={x:0.5}),
                0.4621, places=TestActivations.PLACES)

            # Case 3
            self.assertAlmostEqual(sess.run(f(x), feed_dict={x:-0.25}),
                -0.2449, places=TestActivations.PLACES)

if __name__ == "__main__":
    unittest.main()