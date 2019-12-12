from keras.datasets import mnist
from keras.utils import np_utils
import numpy as np
import h5py

import matplotlib.pylab as plt


def normalization(X):

    return X / 127.5 - 1


def inverse_normalization(X):

    return (X + 1.) / 2.


def load_mnist(image_dim_ordering):

    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    if image_dim_ordering == 'th':
        X_train = X_train.reshape(X_train.shape[0], 1, 28, 28)
        X_test = X_test.reshape(X_test.shape[0], 1, 28, 28)
    else:
        X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
        X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)

    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')

    X_train = normalization(X_train)
    X_test = normalization(X_test)

    nb_classes = len(np.unique(np.hstack((y_train, y_test))))

    Y_train = np_utils.to_categorical(y_train, nb_classes)
    Y_test = np_utils.to_categorical(y_test, nb_classes)

    print X_train.shape, X_test.shape, Y_train.shape, Y_test.shape

    return X_train, Y_train, X_test, Y_test


def load_celebA(img_dim, image_dim_ordering):

    with h5py.File("../../data/processed/CelebA_%s_data.h5" % img_dim, "r") as hf:

        X_real_train = hf["data"][:].astype(np.float32)
        X_real_train = normalization(X_real_train)

        if image_dim_ordering == "tf":
            X_real_train = X_real_train.transpose(0, 2, 3, 1)

        return X_real_train


def gen_batch(X, batch_size):

    while True:
        idx = np.random.choice(X.shape[0], batch_size, replace=False)
        yield X[idx]


def sample_noise(noise_scale, batch_size, noise_dim):

    return np.random.normal(scale=noise_scale, size=(batch_size, noise_dim[0]))


def get_disc_batch(X_real_batch, generator_model, batch_counter, batch_size, noise_dim,
                   noise_scale=0.5, label_smoothing=False, label_flipping=0):

    # Create X_disc: alternatively only generated or real images
    if batch_counter % 2 == 0:
        # Pass noise to the generator
        noise_input = sample_noise(noise_scale, batch_size, noise_dim)
        # Produce an output
        X_disc = generator_model.predict(noise_input)
        y_disc = np.zeros((X_disc.shape[0], 2), dtype=np.uint8)
        y_disc[:, 0] = 1

        if label_flipping > 0:
            p = np.random.binomial(1, label_flipping)
            if p > 0:
                y_disc[:, [0, 1]] = y_disc[:, [1, 0]]

    else:
        X_disc = X_real_batch
        y_disc = np.zeros((X_disc.shape[0], 2), dtype=np.uint8)
        if label_smoothing:
            y_disc[:, 1] = np.random.uniform(low=0.9, high=1, size=y_disc.shape[0])
        else:
            y_disc[:, 1] = 1

        if label_flipping > 0:
            p = np.random.binomial(1, label_flipping)
            if p > 0:
                y_disc[:, [0, 1]] = y_disc[:, [1, 0]]

    return X_disc, y_disc


def get_disc_batch_mixed(X_real_batch, generator_model, batch_counter, batch_size, noise_dim, noise_scale=0.5):

    # Pass noise to the generator
    noise_input = sample_noise(noise_scale, batch_size / 2, noise_dim)
    # Produce an output
    X_disc_noise = generator_model.predict(noise_input)
    y_disc_noise = np.zeros((X_disc_noise.shape[0], 2), dtype=np.uint8)
    y_disc_noise[:, 0] = 1

    X_disc = X_real_batch[:batch_size / 2]
    y_disc = np.zeros((X_disc.shape[0], 2), dtype=np.uint8)
    y_disc[:, 1] = 1

    X_disc = np.concatenate((X_disc, X_disc_noise))
    y_disc = np.concatenate((y_disc, y_disc_noise))

    return X_disc, y_disc


def get_gen_batch(batch_size, noise_dim, noise_scale=0.5):

    X_gen = sample_noise(noise_scale, batch_size, noise_dim)
    y_gen = np.zeros((X_gen.shape[0], 2), dtype=np.uint8)
    y_gen[:, 1] = 1

    return X_gen, y_gen


def plot_generated_batch(X_real, generator_model, batch_size, noise_dim, image_dim_ordering, noise_scale=0.5):

    # Generate images
    X_gen = sample_noise(noise_scale, batch_size, noise_dim)
    X_gen = generator_model.predict(X_gen)

    X_real = inverse_normalization(X_real)
    X_gen = inverse_normalization(X_gen)

    Xg = X_gen[:8]
    Xr = X_real[:8]

    if image_dim_ordering == "tf":
        X = np.concatenate((Xg, Xr), axis=0)
        list_rows = []
        for i in range(int(X.shape[0] / 4)):
            Xr = np.concatenate([X[k] for k in range(4 * i, 4 * (i + 1))], axis=1)
            list_rows.append(Xr)

        Xr = np.concatenate(list_rows, axis=0)

    if image_dim_ordering == "th":
        X = np.concatenate((Xg, Xr), axis=0)
        list_rows = []
        for i in range(int(X.shape[0] / 4)):
            Xr = np.concatenate([X[k] for k in range(4 * i, 4 * (i + 1))], axis=2)
            list_rows.append(Xr)

        Xr = np.concatenate(list_rows, axis=1)
        Xr = Xr.transpose(1,2,0)

    if Xr.shape[-1] == 1:
        plt.imshow(Xr[:, :, 0], cmap="gray")
    else:
        plt.imshow(Xr)
    plt.savefig("../../figures/current_batch.png")
    plt.clf()
    plt.close()
