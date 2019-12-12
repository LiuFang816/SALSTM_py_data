from keras.layers import Flatten, Dense, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.advanced_activations import PReLU

from keras.models import Sequential


def build_cnn(dim, n_classes):
    model = Sequential()

    model.add(Convolution2D(96, 11, 11,
                            subsample=(4, 4),
                            input_shape=(dim, dim, 3),
                            init='glorot_uniform',
                            border_mode='same'))
    model.add(PReLU())
    model.add(MaxPooling2D((3, 3), strides=(2, 2)))

    model.add(Convolution2D(256, 5, 5,
                            subsample=(1, 1),
                            init='glorot_uniform',
                            border_mode='same'))
    model.add(PReLU())
    model.add(MaxPooling2D((3, 3), strides=(2, 2)))

    model.add(Convolution2D(384, 3, 3,
                            subsample=(1, 1),
                            init='glorot_uniform',
                            border_mode='same'))
    model.add(PReLU())

    model.add(Convolution2D(384, 3, 3,
                            subsample=(1, 1),
                            init='glorot_uniform',
                            border_mode='same'))
    model.add(PReLU())

    model.add(Convolution2D(256, 3, 3,
                            subsample=(1, 1),
                            init='glorot_uniform',
                            border_mode='same'))
    model.add(PReLU())
    model.add(MaxPooling2D((3, 3), strides=(2, 2)))

    model.add(Flatten())
    model.add(Dropout(0.5))

    model.add(Dense(2048, init='glorot_uniform'))
    model.add(PReLU())
    model.add(Dropout(0.5))

    model.add(Dense(256, init='glorot_uniform'))
    model.add(PReLU())

    model.add(Dense(n_classes, init='glorot_uniform', activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model
