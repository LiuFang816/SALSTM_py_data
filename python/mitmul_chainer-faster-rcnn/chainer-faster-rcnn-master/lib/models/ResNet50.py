#!/usr/bin/env python
# -*- coding: utf-8 -*-

import chainer
import chainer.functions as F
import chainer.links as L
import math
import numpy as np


class BottleNeckA(chainer.Chain):

    def __init__(self, in_size, ch, out_size, stride=2):
        w = math.sqrt(2)
        super(BottleNeckA, self).__init__(
            conv1=L.Convolution2D(in_size, ch, 1, stride, 0, w, nobias=True),
            bn1=L.BatchNormalization(ch),
            conv2=L.Convolution2D(ch, ch, 3, 1, 1, w, nobias=True),
            bn2=L.BatchNormalization(ch),
            conv3=L.Convolution2D(ch, out_size, 1, 1, 0, w, nobias=True),
            bn3=L.BatchNormalization(out_size),
            conv4=L.Convolution2D(
                in_size, out_size, 1, stride, 0, w, nobias=True),
            bn4=L.BatchNormalization(out_size),
        )

    def __call__(self, x, train):
        h1 = F.relu(self.bn1(self.conv1(x), test=not train))
        h1 = F.relu(self.bn2(self.conv2(h1), test=not train))
        h1 = self.bn3(self.conv3(h1), test=not train)
        h2 = self.bn4(self.conv4(x), test=not train)

        return F.relu(h1 + h2)


class BottleNeckB(chainer.Chain):

    def __init__(self, in_size, ch):
        w = math.sqrt(2)
        super(BottleNeckB, self).__init__(
            conv1=L.Convolution2D(in_size, ch, 1, 1, 0, w, nobias=True),
            bn1=L.BatchNormalization(ch),
            conv2=L.Convolution2D(ch, ch, 3, 1, 1, w, nobias=True),
            bn2=L.BatchNormalization(ch),
            conv3=L.Convolution2D(ch, in_size, 1, 1, 0, w, nobias=True),
            bn3=L.BatchNormalization(in_size),
        )

    def __call__(self, x, train):
        h = F.relu(self.bn1(self.conv1(x), test=not train))
        h = F.relu(self.bn2(self.conv2(h), test=not train))
        h = self.bn3(self.conv3(h), test=not train)

        return F.relu(h + x)


class Block(chainer.Chain):

    def __init__(self, layer, in_size, ch, out_size, stride=2):
        super(Block, self).__init__()
        links = [('a', BottleNeckA(in_size, ch, out_size, stride))]
        for i in range(layer - 1):
            links += [('b_{}'.format(i + 1), BottleNeckB(out_size, ch))]

        for link in links:
            self.add_link(*link)
        self.forward = links

    def __call__(self, x, train):
        for name, _ in self.forward:
            f = getattr(self, name)(x, train)

        return x


class ResNet50(chainer.Chain):

    def __init__(self):
        self.train = True
        w = math.sqrt(2)
        super(ResNet50, self).__init__()
        self.add_link('conv1', L.Convolution2D(3, 64, 7, 2, 3, w, nobias=True))
        self.add_link('bn1', L.BatchNormalization(64))
        self.add_link('res2', Block(3, 64, 64, 256, 1))
        self.add_link('res3', Block(4, 256, 128, 512))
        self.add_link('res4', Block(6, 512, 256, 1024))
        self.add_link('res5', Block(3, 1024, 512, 2048))

    def clear(self):
        self.loss = None
        self.pred = None
        self.pre_rec = None

    def __call__(self, x, t, before_fc=False):
        self.clear()
        h = self.bn1(self.conv1(x), test=not self.train)
        h = F.max_pooling_2d(F.relu(h), 3, stride=2)
        h = self.res2(h, self.train)
        h = self.res3(h, self.train)
        h = self.res4(h, self.train)
        h = self.res5(h, self.train)
        h = F.average_pooling_2d(h, h.data.shape[2], stride=1)
        self.feature = h
        return h
