import sys
# append tinyfacerec to module search path
sys.path.append("..")
# import numpy and matplotlib colormaps
import numpy as np
# import tinyfacerec modules
from tinyfacerec.util import read_images
from tinyfacerec.model import EigenfacesModel

if __name__ == '__main__':

    if len(sys.argv) != 2:
        print "USAGE: example_model_eigenfaces.py </path/to/images>"
        sys.exit()
    
    # read images
    [X,y] = read_images(sys.argv[1])
    # compute the eigenfaces model
    model = EigenfacesModel(X[1:], y[1:])
    # get a prediction for the first observation
    print "expected =", y[0], "/", "predicted =", model.predict(X[0])
