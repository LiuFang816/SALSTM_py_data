import keras.backend as K
import numpy as np


class Bottleneck:
    def __init__(self, model, layer):
        self.fn = K.function([model.layers[0].input, K.learning_phase()], [model.layers[layer].output])

    def predict(self, data_x, batch_size=32, learning_phase=False):
        n_data = len(data_x)
        n_batches = n_data // batch_size + (0 if n_data % batch_size == 0 else 1)

        result = None

        learning_phase = 1 if learning_phase else 0

        for i in range(n_batches):
            batch_x = data_x[i * batch_size:(i + 1) * batch_size]
            batch_y = self.fn([batch_x, 0])[0]

            if result is None:
                result = batch_y
            else:
                result = np.vstack([result, batch_y])

        return result
