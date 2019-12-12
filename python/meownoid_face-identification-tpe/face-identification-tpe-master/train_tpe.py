import itertools

import numpy as np

from cnn import build_cnn
from tpe import build_tpe
from bottleneck import Bottleneck
from identification import get_scores, calc_metrics

from sklearn.decomposition import PCA

n_in = 256
n_out = 256


cnn = build_cnn(227, 266)
cnn.load_weights('data/weights/weights.best.h5')
bottleneck = Bottleneck(cnn, ~1)

train_x, train_y = np.load('./data/train_x.npy'), np.load('./data/train_y.npy')
test_x, test_y = np.load('./data/test_x.npy'), np.load('./data/test_y.npy')

train_x = np.vstack([train_x, test_x])
train_y = np.hstack([train_y, test_y])

dev_x = np.load('./data/dev_x.npy')
dev_protocol = np.load('./data/dev_protocol.npy')

train_emb = bottleneck.predict(train_x, batch_size=256)
dev_emb = bottleneck.predict(dev_x, batch_size=256)

del train_x

pca = PCA(n_out)
pca.fit(train_emb)
W_pca = pca.components_

tpe, tpe_pred = build_tpe(n_in, n_out, W_pca.T)
# tpe.load_weights('data/weights/weights.tpe.mineer.h5')

train_y = np.array(train_y)
subjects = list(set(train_y))

anchors_inds = []
positives_inds = []
labels = []

for subj in subjects:
    mask = train_y == subj
    inds = np.where(mask)[0]
    for a, p in itertools.permutations(inds, 2):
        anchors_inds.append(a)
        positives_inds.append(p)
        labels.append(subj)

anchors = train_emb[anchors_inds]
positives = train_emb[positives_inds]
n_anchors = len(anchors_inds)


NB_EPOCH = 5000
COLD_START = NB_EPOCH
BATCH_SIZE = 4
BIG_BATCH_SIZE = 512


inds = np.arange(n_anchors)


def get_batch(hard=False):
    batch_inds = np.random.choice(inds, size=BIG_BATCH_SIZE, replace=False)

    train_emb2 = tpe_pred.predict(train_emb, batch_size=1024)
    scores = train_emb2 @ train_emb2.T
    negative_inds = []

    for i in batch_inds:
        label = labels[i]
        mask = train_y == label
        if hard:
            negative_inds.append(np.ma.array(scores[label], mask=mask).argmax())
        else:
            negative_inds.append(np.random.choice(np.where(np.logical_not(mask))[0], size=1)[0])

    return anchors[batch_inds], positives[batch_inds], train_emb[negative_inds]


def test():
    dev_emb2 = tpe_pred.predict(dev_emb)
    tsc, isc = get_scores(dev_emb2, dev_protocol)
    eer, _, _, _ = calc_metrics(tsc, isc)
    return eer


z = np.zeros((BIG_BATCH_SIZE,))

mineer = float('inf')

for e in range(NB_EPOCH):
    print('epoch: {}'.format(e))
    a, p, n = get_batch(e > COLD_START)
    tpe.fit([a, p, n], z, batch_size=BATCH_SIZE, nb_epoch=1)
    eer = test()
    print('EER: {:.2f}'.format(eer * 100))
    if eer < mineer:
        mineer = eer
        tpe.save_weights('data/weights/weights.tpe.mineer.h5')
