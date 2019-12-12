from keras.layers import Dense, Lambda, Input, merge
from keras.models import Model, Sequential
from keras.optimizers import SGD

import keras.backend as K

import numpy as np


def triplet_loss(y_true, y_pred):
    return -K.mean(K.log(K.sigmoid(y_pred)))


def triplet_merge(inputs):
    a, p, n = inputs

    return K.sum(a * (p - n), axis=1)


def triplet_merge_shape(input_shapes):
    return (input_shapes[0][0], 1)


def build_tpe(n_in, n_out, W_pca=None):
    a = Input(shape=(n_in,))
    p = Input(shape=(n_in,))
    n = Input(shape=(n_in,))

    if W_pca is None:
        W_pca = np.zeros((n_in, n_out))

    base_model = Sequential()
    base_model.add(Dense(n_out, input_dim=n_in, bias=False, weights=[W_pca], activation='linear'))
    base_model.add(Lambda(lambda x: K.l2_normalize(x, axis=1)))

    # base_model = Sequential()
    # base_model.add(Dense(178, input_dim=n_in, bias=True, activation='relu'))
    # base_model.add(Dense(n_out, bias=True, activation='tanh'))
    # base_model.add(Lambda(lambda x: K.l2_normalize(x, axis=1)))

    a_emb = base_model(a)
    p_emb = base_model(p)
    n_emb = base_model(n)

    e = merge([a_emb, p_emb, n_emb], mode=triplet_merge, output_shape=triplet_merge_shape)

    model = Model(input=[a, p, n], output=e)
    predict = Model(input=a, output=a_emb)

    model.compile(loss=triplet_loss, optimizer='rmsprop')

    return model, predict
