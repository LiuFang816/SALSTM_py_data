#! /usr/bin/python
# -*- coding: utf8 -*-




from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD, Adam, RMSprop
from keras.utils import np_utils
import time

nb_classes = 10

# the data, shuffled and split between train and test sets
(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.reshape(60000, 784)
X_test = X_test.reshape(10000, 784)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

X_train, X_val = X_train[:-10000], X_train[-10000:]
Y_train, Y_val = Y_train[:-10000], Y_train[-10000:]

print(X_train.shape[0], 'train samples')
print(X_val.shape[0], 'test samples')
print(X_test.shape[0], 'test samples')


print(Y_train.shape, Y_test.shape)

model = Sequential()
model.add(Dropout(0.2, input_shape=(784,)))
model.add(Dense(800))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(800))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(10))
model.add(Activation('softmax'))

model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer=Adam(),
              metrics=['accuracy'])

start_time = time.time()
history = model.fit(X_train, Y_train,
                    batch_size=500, nb_epoch=200,
                    verbose=1, validation_data=(X_val, Y_val)) 
print("Total training time: %fs" % (time.time() - start_time))
score = model.evaluate(X_test, Y_test, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])
