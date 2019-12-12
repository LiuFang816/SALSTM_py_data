# -*- coding: utf-8 -*-
"""
=============================
Hyperalign a list of arrays and create an animated plot
=============================

The sample data is a list of 2D arrays, where each array is fMRI brain activity
from one subject.  The rows are timepoints and the columns are neural
'features'.  First, the matrices are hyperaligned using hyp.tools.align.
"""

# Code source: Andrew Heusser
# License: MIT

import hypertools as hyp
import scipy.io as sio
import numpy as np

data=sio.loadmat('sample_data/weights.mat')
w=data['weights'][0]
w = [i for i in w]
aligned_w = hyp.tools.align(w)

w1 = np.mean(aligned_w[:17],0)
w2 = np.mean(aligned_w[18:],0)

hyp.plot([w1,w2],animate=True, duration=100)
