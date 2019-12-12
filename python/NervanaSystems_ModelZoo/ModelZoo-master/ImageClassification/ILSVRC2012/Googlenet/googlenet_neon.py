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
Googlenet V1 implementation
"""

import os

from neon.util.argparser import NeonArgparser
from neon.layers import Conv, Pooling, MergeBroadcast, BranchNode
from neon.layers import Affine, SingleOutputTree, Dropout
from neon.layers import GeneralizedCost, Multicost
from neon.initializers import Constant, Xavier
from neon.backends import gen_backend
from neon.optimizers import GradientDescentMomentum, MultiOptimizer
from neon.transforms import Rectlin, Softmax, CrossEntropyMulti, TopKMisclassification
from neon.models import Model
from neon.data import ImageLoader

parser = NeonArgparser(__doc__)
parser.add_argument('--subset_pct', type=float, default=100,
                    help='subset of training dataset to use (percentage)')
parser.add_argument('--test_only', action='store_true',
                    help='skip fitting - evaluate metrics on trained model weights')
args = parser.parse_args()

# setup data provider
img_set_options = dict(repo_dir=args.data_dir, inner_size=224,
                       subset_pct=args.subset_pct)
test = ImageLoader(set_name='validation', scale_range=(256, 256),
                   do_transforms=False, **img_set_options)

init1 = Xavier(local=False)
initx = Xavier(local=True)
bias = Constant(val=0.20)
relu = Rectlin()

common = dict(activation=relu, init=initx, bias=bias)
commonp1 = dict(activation=relu, init=initx, bias=bias, padding=1)
commonp2 = dict(activation=relu, init=initx, bias=bias, padding=2)
pool3s1p1 = dict(fshape=3, padding=1, strides=1)
pool3s2p1 = dict(fshape=3, padding=1, strides=2, op='max')


def inception(kvals):
    (p1, p2, p3, p4) = kvals

    branch1 = [Conv((1, 1, p1[0]), **common)]
    branch2 = [Conv((1, 1, p2[0]), **common),
               Conv((3, 3, p2[1]), **commonp1)]
    branch3 = [Conv((1, 1, p3[0]), **common),
               Conv((5, 5, p3[1]), **commonp2)]
    branch4 = [Pooling(op="max", **pool3s1p1),
               Conv((1, 1, p4[0]), **common)]
    return MergeBroadcast(layers=[branch1, branch2, branch3, branch4], merge="depth")

def main_branch(branch_nodes):
    return [Conv((7, 7, 64), padding=3, strides=2, **common),
            Pooling(**pool3s2p1),
            Conv((1, 1, 64), **common),
            Conv((3, 3, 192), **commonp1),
            Pooling(**pool3s2p1),
            inception([(64, ), (96, 128), (16, 32), (32, )]),
            inception([(128,), (128, 192), (32, 96), (64, )]),
            Pooling(**pool3s2p1),
            inception([(192,), (96, 208), (16, 48), (64, )]),
            branch_nodes[0],
            inception([(160,), (112, 224), (24, 64), (64, )]),
            inception([(128,), (128, 256), (24, 64), (64, )]),
            inception([(112,), (144, 288), (32, 64), (64, )]),
            branch_nodes[1],
            inception([(256,), (160, 320), (32, 128), (128,)]),
            Pooling(**pool3s2p1),
            inception([(256,), (160, 320), (32, 128), (128,)]),
            inception([(384,), (192, 384), (48, 128), (128,)]),
            Pooling(fshape=7, strides=1, op="avg"),
            Affine(nout=1000, init=init1, activation=Softmax(), bias=Constant(0))]


def aux_branch(bnode):
    return [bnode,
            Pooling(fshape=5, strides=3, op="avg"),
            Conv((1, 1, 128), **common),
            Affine(nout=1024, init=init1, activation=relu, bias=bias),
            Dropout(keep=0.3),
            Affine(nout=1000, init=init1, activation=Softmax(), bias=Constant(0))]


# Now construct the model
branch_nodes = [BranchNode(name='branch' + str(i)) for i in range(2)]
main1 = main_branch(branch_nodes)
aux1 = aux_branch(branch_nodes[0])
aux2 = aux_branch(branch_nodes[1])

model = Model(layers=SingleOutputTree([main1, aux1, aux2], alphas=[1.0, 0.3, 0.3]))

valmetric = TopKMisclassification(k=5)

# dummy optimizer for benchmarking
# training implementation coming soon
opt_gdm = GradientDescentMomentum(0.0, 0.0)
opt_biases = GradientDescentMomentum(0.0, 0.0)
opt = MultiOptimizer({'default': opt_gdm, 'Bias': opt_biases})

# setup cost function as CrossEntropy
cost = Multicost(costs=[GeneralizedCost(costfunc=CrossEntropyMulti()),
                        GeneralizedCost(costfunc=CrossEntropyMulti()),
                        GeneralizedCost(costfunc=CrossEntropyMulti())],
                 weights=[1, 0., 0.])  # We only want to consider the CE of the main path

assert os.path.exists(args.model_file), 'script requires the trained weights file'
model.load_params(args.model_file)
model.initialize(test, cost)


print 'running speed benchmark...'
model.benchmark(test, cost, opt)

print '\nCalculating performance on validation set...'
test.reset()
mets = model.eval(test, metric=valmetric)
print 'Validation set metrics:'
print 'LogLoss: %.2f, Accuracy: %.1f %% (Top-1), %.1f %% (Top-5)' % (mets[0],
                                                                     (1.0-mets[1])*100,
                                                                     (1.0-mets[2])*100)
