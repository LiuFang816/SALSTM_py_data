#!/usr/bin/env python
# ----------------------------------------------------------------------------
# Copyright 2016 Nervana Systems Inc.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ----------------------------------------------------------------------------
from neon.util.argparser import NeonArgparser
from neon.initializers import Kaiming, IdentityInit
from neon.layers import Conv, Pooling, GeneralizedCost, Affine, Activation
from neon.layers import MergeSum, SkipNode
from neon.optimizers import GradientDescentMomentum, Schedule
from neon.transforms import Rectlin, Softmax, CrossEntropyMulti, Misclassification
from neon.models import Model
from neon.data import ImageLoader, ImageParams, DataLoader
from neon.callbacks.callbacks import Callbacks

import os

# parse the command line arguments (generates the backend)
parser = NeonArgparser(__doc__)
parser.add_argument('--depth', type=int, default=9,
                    help='depth of each stage (network depth will be 6n+2)')
args = parser.parse_args()


def extract_images(out_dir, padded_size):
    '''
    Save CIFAR-10 dataset as PNG files
    '''
    import numpy as np
    from neon.data import load_cifar10
    from PIL import Image
    dataset = dict()
    dataset['train'], dataset['val'], _ = load_cifar10(out_dir, normalize=False)
    pad_size = (padded_size - 32) // 2 if padded_size > 32 else 0
    pad_width = ((0, 0), (pad_size, pad_size), (pad_size, pad_size))

    for setn in ('train', 'val'):
        data, labels = dataset[setn]

        img_dir = os.path.join(out_dir, setn)
        ulabels = np.unique(labels)
        for ulabel in ulabels:
            subdir = os.path.join(img_dir, str(ulabel))
            if not os.path.exists(subdir):
                os.makedirs(subdir)

        for idx in range(data.shape[0]):
            im = np.pad(data[idx].reshape((3, 32, 32)), pad_width, mode='mean')
            im = np.uint8(np.transpose(im, axes=[1, 2, 0]).copy())
            im = Image.fromarray(im)
            path = os.path.join(img_dir, str(labels[idx][0]), str(idx) + '.png')
            im.save(path, format='PNG')

# setup data provider
train_dir = os.path.join(args.data_dir, 'train')
test_dir = os.path.join(args.data_dir, 'val')
if not (os.path.exists(train_dir) and os.path.exists(test_dir)):
    extract_images(args.data_dir, 40)

# setup data provider
shape = dict(channel_count=3, height=32, width=32)
train_params = ImageParams(center=False, flip=True, **shape)
test_params = ImageParams(**shape)
common = dict(target_size=1, nclasses=10)

train = DataLoader(set_name='train', repo_dir=train_dir, media_params=train_params,
                   shuffle=True, **common)
test = DataLoader(set_name='val', repo_dir=test_dir, media_params=test_params, **common)


def conv_params(fsize, nfm, stride=1, relu=True):
    return dict(fshape=(fsize, fsize, nfm), strides=stride, padding=(1 if fsize > 1 else 0),
                activation=(Rectlin() if relu else None),
                init=Kaiming(local=True),
                batch_norm=True)


def id_params(nfm):
    return dict(fshape=(1, 1, nfm), strides=2, padding=0, activation=None, init=IdentityInit())


def module_factory(nfm, stride=1):
    mainpath = [Conv(**conv_params(3, nfm, stride=stride)),
                Conv(**conv_params(3, nfm, relu=False))]
    sidepath = [SkipNode() if stride == 1 else Conv(**id_params(nfm))]
    module = [MergeSum([mainpath, sidepath]),
              Activation(Rectlin())]
    return module

# Structure of the deep residual part of the network:
# args.depth modules of 2 convolutional layers each at feature map depths of 16, 32, 64
nfms = [2**(stage + 4) for stage in sorted(range(3) * args.depth)]
strides = [1] + [1 if cur == prev else 2 for cur, prev in zip(nfms[1:], nfms[:-1])]

# Now construct the network
layers = [Conv(**conv_params(3, 16))]
for nfm, stride in zip(nfms, strides):
    layers.append(module_factory(nfm, stride))
layers.append(Pooling(8, op='avg'))
layers.append(Affine(nout=10, init=Kaiming(local=False), batch_norm=True, activation=Softmax()))

model = Model(layers=layers)
opt = GradientDescentMomentum(0.1, 0.9, wdecay=0.0001,
                              schedule=Schedule([90, 123], 0.1))

# configure callbacks
callbacks = Callbacks(model, eval_set=test, metric=Misclassification(), **args.callback_args)
cost = GeneralizedCost(costfunc=CrossEntropyMulti())

model.fit(train, optimizer=opt, num_epochs=args.epochs, cost=cost, callbacks=callbacks)
