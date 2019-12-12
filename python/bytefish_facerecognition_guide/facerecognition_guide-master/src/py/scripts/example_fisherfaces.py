import sys
# append tinyfacerec to module search path
sys.path.append("..")
# import numpy and matplotlib colormaps
import numpy as np
# import tinyfacerec modules
from tinyfacerec.subspace import fisherfaces
from tinyfacerec.util import normalize, asRowMatrix, read_images
from tinyfacerec.visual import subplot

if __name__ == '__main__':

    if len(sys.argv) != 2:
        print "USAGE: example_fisherfaces.py </path/to/images>"
        sys.exit()

    # read images
    [X,y] = read_images(sys.argv[1])
    # perform a full pca
    [D, W, mu] = fisherfaces(asRowMatrix(X), y)
    #import colormaps
    import matplotlib.cm as cm
    # turn the first (at most) 16 eigenvectors into grayscale
    # images (note: eigenvectors are stored by column!)
    E = []
    for i in xrange(min(W.shape[1], 16)):
	    e = W[:,i].reshape(X[0].shape)
	    E.append(normalize(e,0,255))
    # plot them and store the plot to "python_fisherfaces_fisherfaces.pdf"
    subplot(title="Fisherfaces AT&T Facedatabase", images=E, rows=4, cols=4, sptitle="Fisherface", colormap=cm.jet, filename="python_fisherfaces_fisherfaces.png")

    from tinyfacerec.subspace import project, reconstruct

    E = []
    for i in xrange(min(W.shape[1], 16)):
	    e = W[:,i].reshape(-1,1)
	    P = project(e, X[0].reshape(1,-1), mu)
	    R = reconstruct(e, P, mu)
	    # reshape and append to plots
	    R = R.reshape(X[0].shape)
	    E.append(normalize(R,0,255))
    # plot them and store the plot to "python_reconstruction.pdf"
    subplot(title="Fisherfaces Reconstruction Yale FDB", images=E, rows=4, cols=4, sptitle="Fisherface", colormap=cm.gray, filename="python_fisherfaces_reconstruction.png")
