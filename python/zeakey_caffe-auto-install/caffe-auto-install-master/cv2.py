import numpy as np 
import scipy.misc as misc

def imread(imdir):
    im=misc.imread(imdir)
    return im

def imwrite(fn, im):
    misc.imsave(fn, im)

def imresize(im, scale):
    im1=misc.imresize(im, scale)
    return im1
