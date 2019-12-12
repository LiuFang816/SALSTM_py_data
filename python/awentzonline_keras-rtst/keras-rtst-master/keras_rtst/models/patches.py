from itertools import product

import numpy as np
from keras import backend as K
from sklearn.feature_extraction.image import reconstruct_from_patches_2d


def make_patches(x, patch_size, patch_stride):
    '''Break image `x` up into a bunch of patches.'''
    from theano.tensor.nnet.neighbours import images2neibs
    patches = images2neibs(x,
        (patch_size, patch_size), (patch_stride, patch_stride),
        mode='valid')
    # neibs are sorted per-channel
    patches = K.reshape(patches, (K.shape(x)[1], K.shape(patches)[0] // K.shape(x)[1], patch_size, patch_size))
    patches = K.permute_dimensions(patches, (1, 0, 2, 3))
    patches_norm = K.sqrt(K.sum(K.square(patches), axis=(1,2,3), keepdims=True))
    return patches, patches_norm


def combine_patches(patches, out_shape):
    '''Reconstruct an image from these `patches`'''
    patches = patches.transpose(0, 2, 3, 1)
    recon = reconstruct_from_patches_2d(patches, out_shape)
    return recon.transpose(2, 0, 1)


def find_patch_matches(a, a_norm, b):
    '''For each patch in A, find the best matching patch in B'''
    # we want cross-correlation here so flip the kernels
    convs = K.conv2d(a, b[:, :, ::-1, ::-1], border_mode='valid')
    argmax = K.argmax(convs / a_norm, axis=1)
    return argmax
