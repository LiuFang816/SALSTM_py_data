import numpy as np
import matplotlib.pyplot as plt

from cnn import build_cnn
from bottleneck import Bottleneck

from identification import get_scores, calc_metrics

WEIGHTS_DIR = './data/weights/'

BATCH_SIZE = 32

dev_x = np.load('data/dev_x.npy')

model = build_cnn(227, 266)

weights_to_load = WEIGHTS_DIR + 'weights.best.h5'
model.load_weights(weights_to_load)

bottleneck = Bottleneck(model, ~1)
dev_y = bottleneck.predict(dev_x, batch_size=BATCH_SIZE)

protocol = np.load('data/dev_protocol.npy')
tsc, isc = get_scores(dev_y, protocol)
eer, fars, frrs, dists = calc_metrics(tsc, isc)

print('EER: {}'.format(eer * 100))

plt.figure()
plt.hist(tsc, 20, color='g', normed=True, alpha=0.3)
plt.hist(isc, 20, color='r', normed=True, alpha=0.3)

plt.figure()
plt.loglog(fars, frrs)
plt.show()
