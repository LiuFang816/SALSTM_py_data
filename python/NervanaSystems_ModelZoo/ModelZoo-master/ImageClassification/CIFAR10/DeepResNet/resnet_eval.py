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
import os

from neon.util.argparser import NeonArgparser
from neon.util.persist import load_obj
from neon.transforms import Misclassification, CrossEntropyMulti
from neon.optimizers import GradientDescentMomentum
from neon.layers import GeneralizedCost
from neon.models import Model
from neon.data import DataLoader, ImageParams

# parse the command line arguments (generates the backend)
parser = NeonArgparser(__doc__)
args = parser.parse_args()

# setup data provider
test_dir = os.path.join(args.data_dir, 'val')
shape = dict(channel_count=3, height=32, width=32)
test_params = ImageParams(center=True, flip=False, **shape)
common = dict(target_size=1, nclasses=10)
test_set = DataLoader(set_name='val', repo_dir=test_dir, media_params=test_params, **common)

model = Model(load_obj(args.model_file))
cost = GeneralizedCost(costfunc=CrossEntropyMulti())
opt = GradientDescentMomentum(0.1, 0.9, wdecay=0.0001)
model.initialize(test_set, cost=cost)

acc = 1.0 - model.eval(test_set, metric=Misclassification())[0]
print 'Accuracy: %.1f %% (Top-1)' % (acc*100.0)

model.benchmark(test_set, cost=cost, optimizer=opt)
