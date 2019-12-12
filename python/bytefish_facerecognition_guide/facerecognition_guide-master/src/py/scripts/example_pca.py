import sys
# append tinyfacerec to module search path
sys.path.append("..")
# import numpy and matplotlib colormaps
import numpy as np
# import tinyfacerec modules
from tinyfacerec.subspace import pca
from tinyfacerec.util import normalize, asRowMatrix, read_images
from tinyfacerec.visual import subplot

# read images
[X,y] = read_images("/home/philipp/facerec/data/at")
# perform a full pca
[D, W, mu] = pca(asRowMatrix(X), y)

import matplotlib.cm as cm

# turn the first (at most) 16 eigenvectors into grayscale
# images (note: eigenvectors are stored by column!)
E = []
for i in xrange(min(len(X), 16)):
	e = W[:,i].reshape(X[0].shape)
	E.append(normalize(e,0,255))
# plot them and store the plot to "python_eigenfaces.pdf"
subplot(title="Eigenfaces AT&T Facedatabase", images=E, rows=4, cols=4, sptitle="Eigenface", colormap=cm.jet, filename="python_pca_eigenfaces.pdf")

from tinyfacerec.subspace import project, reconstruct

# reconstruction steps
steps=[i for i in xrange(10, min(len(X), 320), 20)]
E = []
for i in xrange(min(len(steps), 16)):
	numEvs = steps[i]
	P = project(W[:,0:numEvs], X[0].reshape(1,-1), mu)
	R = reconstruct(W[:,0:numEvs], P, mu)
	# reshape and append to plots
	R = R.reshape(X[0].shape)
	E.append(normalize(R,0,255))
# plot them and store the plot to "python_reconstruction.pdf"
subplot(title="Reconstruction AT&T Facedatabase", images=E, rows=4, cols=4, sptitle="Eigenvectors", sptitles=steps, colormap=cm.gray, filename="python_pca_reconstruction.pdf")
