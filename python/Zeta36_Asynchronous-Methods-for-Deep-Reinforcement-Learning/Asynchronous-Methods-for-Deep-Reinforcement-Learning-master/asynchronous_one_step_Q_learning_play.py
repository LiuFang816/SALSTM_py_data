#!/usr/bin/env python
import tensorflow as tf
import cv2
import sys
sys.path.append("Wrapped Game Code/")
import pong_fun as game # whichever is imported "as game" will be used
import dummy_game #as game
import tetris_fun #as game
import numpy as np

GAME = 'pong' # the name of the game being played for log files
ACTIONS = 3 # number of valid actions

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev = 0.01)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.01, shape = shape)
    return tf.Variable(initial)

def conv2d(x, W, stride):
    return tf.nn.conv2d(x, W, strides = [1, stride, stride, 1], padding = "SAME")

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = "SAME")

def createNetwork():
    # network weights
    W_conv1 = weight_variable([8, 8, 4, 32])
    b_conv1 = bias_variable([32])

    W_conv2 = weight_variable([4, 4, 32, 64])
    b_conv2 = bias_variable([64])

    W_conv3 = weight_variable([3, 3, 64, 64])
    b_conv3 = bias_variable([64])

    W_fc1 = weight_variable([256, 256])
    b_fc1 = bias_variable([256])

    W_fc2 = weight_variable([256, ACTIONS])
    b_fc2 = bias_variable([ACTIONS])

    # input layer
    s = tf.placeholder("float", [None, 80, 80, 4])

    # hidden layers
    h_conv1 = tf.nn.relu(conv2d(s, W_conv1, 4) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2, 2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

    h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3, 1) + b_conv3)
    h_pool3 = max_pool_2x2(h_conv3)

    h_pool3_flat = tf.reshape(h_pool3, [-1, 256])

    h_fc1 = tf.nn.relu(tf.matmul(h_pool3_flat, W_fc1) + b_fc1)

    # readout layer
    readout = tf.matmul(h_fc1, W_fc2) + b_fc2

    return s, readout, W_conv1, b_conv1, W_conv2, b_conv2, W_conv3, b_conv3, W_fc1, b_fc1, W_fc2, b_fc2
    	
def playGame(sess):
    # open up a game state to communicate with emulator
    game_state = game.GameState()

    score = 0

    # get the first state by doing nothing and preprocess the image to 80x80x4
    x_t, r_0, terminal = game_state.frame_step([1, 0, 0])
    x_t = cv2.cvtColor(cv2.resize(x_t, (80, 80)), cv2.COLOR_BGR2GRAY)
    s_t = np.stack((x_t, x_t, x_t, x_t), axis = 2)
    aux_s = s_t

    t = 0
    while not terminal:

        # choose an action
        readout_t = O_readout.eval(session = sess, feed_dict = {s : [s_t]})[0]
        a_t = np.zeros([ACTIONS])

        action_index = np.argmax(readout_t)
        a_t[action_index] = 1

        # run the selected action and observe next state and reward
        x_t1_col, r_t, terminal = game_state.frame_step(a_t)
        x_t1 = cv2.cvtColor(cv2.resize(x_t1_col, (80, 80)), cv2.COLOR_BGR2GRAY)
        x_t1 = np.reshape(x_t1, (80, 80, 1))
        aux_s = np.delete(s_t, 0, axis = 2)
        s_t1 = np.append(aux_s, x_t1, axis = 2)

        # update state and score
        s_t = s_t1
        t += 1
        score += r_t

        print "TIMESTEP", t, "/ ACTION", action_index, "/ REWARD", r_t
        print readout_t

    # Print final score
    print "FINAL SCORE", score


# We restore the O network
s, O_readout, W_conv1, b_conv1, W_conv2, b_conv2, W_conv3, b_conv3, W_fc1, b_fc1, W_fc2, b_fc2 = createNetwork()

# Initialize session a variables
sess = tf.InteractiveSession()

if __name__ == "__main__":
    #Restore trained network
    saver = tf.train.Saver()
    checkpoint = tf.train.get_checkpoint_state("save_networks_asyn")
    if checkpoint and checkpoint.model_checkpoint_path:
        saver.restore(sess, checkpoint.model_checkpoint_path)
        print "Successfully loaded:", checkpoint.model_checkpoint_path

    #Play a game to test score
    for i in range(25):
        playGame(sess)
