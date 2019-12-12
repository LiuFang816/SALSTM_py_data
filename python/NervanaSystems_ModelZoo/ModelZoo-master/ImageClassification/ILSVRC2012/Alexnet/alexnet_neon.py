#!/usr/bin/env python
# ----------------------------------------------------------------------------
# Copyright 2015 Nervana Systems Inc.
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
"""
Caffenet implementation:
An Alexnet like model adapted to neon.  Does not include the weight grouping.

See:
    http://dl.caffe.berkeleyvision.org/bvlc_reference_caffenet.caffemodel

To run the complete training for 60 epochs
    alexnet_neon.py -e 60 -eval 1 -s <save-path> -w <path-to-saved-batches>

To load a pretrained model and run it on the validation set:
    alexnet_neon.py -w <path-to-saved-batches> --test_only \
            --model_file <saved weights file>
"""

from neon.util.argparser import NeonArgparser
from neon.initializers import Constant, Gaussian
from neon.layers import Conv, Dropout, Pooling, GeneralizedCost, Affine, LRN
from neon.optimizers import GradientDescentMomentum, MultiOptimizer, Schedule
from neon.transforms import Rectlin, Softmax, CrossEntropyMulti, TopKMisclassification
from neon.models import Model
from neon.data import ImageLoader
from neon.callbacks.callbacks import Callbacks

# parse the command line arguments (generates the backend)
parser = NeonArgparser(__doc__)
parser.add_argument('--subset_pct', type=float, default=100,
                    help='subset of training dataset to use (percentage)')
parser.add_argument('--test_only', action='store_true',
                    help='skip fitting - evaluate metrics on trained model weights')
args = parser.parse_args()

if args.test_only:
    if args.model_file is None:
        raise ValueError('To test model, trained weights need to be provided')

# setup data provider
img_set_options = dict(repo_dir=args.data_dir,
                       inner_size=224,
                       subset_pct=args.subset_pct)
train = ImageLoader(set_name='train', scale_range=(256, 256),
                    shuffle=True, **img_set_options)
test = ImageLoader(set_name='validation', scale_range=(256, 256),
                   do_transforms=False, **img_set_options)

init_g1 = Gaussian(scale=0.01)
init_g2 = Gaussian(scale=0.005)

relu = Rectlin()

layers = []
layers.append(Conv((11, 11, 96), padding=0, strides=4,
                   init=init_g1, bias=Constant(0), activation=relu, name='conv1'))

layers.append(Pooling(3, strides=2, name='pool1'))
layers.append(LRN(5, ascale=0.0001, bpower=0.75, name='norm1'))
layers.append(Conv((5, 5, 256), padding=2, init=init_g1,
                    bias=Constant(1.0), activation=relu, name='conv2'))

layers.append(Pooling(3, strides=2, name='pool2'))
layers.append(LRN(5, ascale=0.0001, bpower=0.75, name='norm2'))
layers.append(Conv((3, 3, 384), padding=1, init=init_g1, bias=Constant(0),
                    activation=relu, name='conv3'))

layers.append(Conv((3, 3, 384), padding=1, init=init_g1, bias=Constant(1.0),
                    activation=relu, name='conv4'))

layers.append(Conv((3, 3, 256), padding=1, init=init_g1, bias=Constant(1.0),
                    activation=relu, name='conv5'))

layers.append(Pooling(3, strides=2, name='pool5'))
layers.append(Affine(nout=4096, init=init_g2, bias=Constant(1.0),
                      activation=relu, name='fc6'))

layers.append(Dropout(keep=0.5, name='drop6'))
layers.append(Affine(nout=4096, init=init_g2, bias=Constant(1.0),
                      activation=relu, name='fc7'))

layers.append(Dropout(keep=0.5, name='drop7'))
layers.append(Affine(nout=1000, init=init_g1, bias=Constant(0.0),
                      activation=Softmax(), name='fc8'))

model = Model(layers=layers)

# scale LR by 0.1 every 20 epochs (this assumes batch_size = 256)
weight_sched = Schedule(20, 0.1)
opt_gdm = GradientDescentMomentum(0.01, 0.9, wdecay=0.0005, schedule=weight_sched)
opt_biases = GradientDescentMomentum(0.02, 0.9, schedule=weight_sched)
opt = MultiOptimizer({'default': opt_gdm, 'Bias': opt_biases})

# configure callbacks
valmetric = TopKMisclassification(k=5)
callbacks = Callbacks(model, eval_set=test, metric=valmetric, **args.callback_args)

if args.model_file is not None:
    model.load_params(args.model_file)
if not args.test_only:
    cost = GeneralizedCost(costfunc=CrossEntropyMulti())
    model.fit(train, optimizer=opt, num_epochs=args.epochs, cost=cost, callbacks=callbacks)

mets = model.eval(test, metric=valmetric)
print 'Validation set metrics:'
print 'LogLoss: %.2f, Accuracy: %.1f %% (Top-1), %.1f %% (Top-5)' % (mets[0],
                                                                     (1.0-mets[1])*100,
                                                                     (1.0-mets[2])*100)
