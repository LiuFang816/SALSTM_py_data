"""
Copyright (c) 2016 Rodrigo Baravalle
All rights reserved.
Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:
1. Redistributions of source code must retain the above copyright
notice, this list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in the
documentation and/or other materials provided with the distribution.
3. The name of the author may not be used to endorse or promote products
derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE AUTHOR ``AS IS'' AND ANY EXPRESS OR
IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY DIRECT, INDIRECT,
INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT
NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF
THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""


from Algorithm import *
import numpy as np
from math import log10
import scipy.signal
import scipy.io as sio

from MFS_3D import *



class Local_MFS_3D (Algorithm):

    """
    :3D implementation of Local MFS through holder exponents f(alpha)
    :Several MFS are computed on a single domain, from which then
    :a set of operations produces features
    :version: 1.0
    :author: Rodrigo Baravalle
    """

    def __init__(self):
        pass

    def setDef(self, ind, f, ite, filename, file_mask, params):

        # parameters: ind -> determines how many levels are used when computing the density 
        #                          choose 1 for using  directly the image measurement im or
        #                          >= 6 for computing the density of im (quite stable for >=5)      
        #              f ----> determines the dimension of  MFS vector
        #              ite  ---> determines how many levels are used when computing MFS for each 

        self.ind_num = ind      # number of pixels for averaging
        self.f_num = f          # window
        self.ite_num = ite
        self.filename = filename
        self.file_mask = file_mask
        self.params = params



    def gauss_kern(self,size_x, size_y, size_z):
        """ Returns a normalized 3D gauss kernel array for convolutions """
        m = np.float32(size_x)
        n = np.float32(size_y)
        o = np.float32(size_z)
        sigma = 2;     # ???
        if(size_x <= 3): sigma = 1.5;
        if(size_x == 5): sigma = 2.5;
        z, y, x = np.mgrid[-(m-1)/2:(m-1)/2+1, -(n-1)/2:(n-1)/2+1, -(o-1)/2:(o-1)/2+1]

        b = 2*(sigma**2)
        square = lambda i : i**2
        fm = lambda i: map(square, i)

        x2 = map(fm, x)
        y2 = map(fm, y)
        z2 = map(fm, z)

        g = np.sum([x2, y2, z2], axis=0).astype(np.float32)
        g = np.exp(g).astype(np.float32)
        return g / g.sum()

    def determine_threshold(self, arr):
        # compute histogram of values
        bins = range(np.min(arr), np.max(arr) + 1)

        h = np.histogram(arr, bins=bins)

        threshold = np.min(arr)

        # get x% of mass -> threshold
        assert (len(arr.shape) == 3)

        total_pixels = arr.shape[0] * arr.shape[1] * arr.shape[2]

        for i in range(len(bins) + 1):
            # compute sum of h(x) from x = 0 to x = i
            partial_sum_vector = np.cumsum(h[0][: (i + 1)])
            partial_sum = partial_sum_vector[len(partial_sum_vector) - 1]

            percentage = (float)(partial_sum) / (float)(total_pixels)

            if percentage > 0.75:
                threshold = np.min(arr) + i
                break

        return threshold



    def openMatlab(self, name, filename, greyscale):

        import scipy.io as sio
        arr = np.array(sio.loadmat(filename)[name]).astype(np.int32)

        if greyscale:
            return arr

        if name == "S":
            threshold = self.determine_threshold(arr)

            arr = arr > threshold

            a_v = arr.cumsum()

            print "Amount of white pixels: ", a_v[len(a_v) - 1]

        # debug - to see the spongious structure
        # plt.imshow((arr[:,:,50]), cmap=plt.gray())
        # plt.show()

        return arr

    def getFDs(self):
        """
        @param string filename : volume location
        @param string file_mask : mask volume location
        @return [float] : 3D (local) multi fractal dimentions
        @author: Rodrigo Baravalle.
        """

        # data is a 3D grayscale volume
        data = self.openMatlab('S', self.filename, True)
        data_mask = self.openMatlab('M', self.file_mask, True)

        # use the base 3D MFS with the same parameters
        # fix me - parameter handling
        base_MFS = MFS_3D()
        base_MFS.setDef(self.ind_num, self.f_num, self.ite_num,
                        self.filename, self.file_mask, self.params)

        # Masking
        data = data * (data_mask > 0)

        # Other multifractal measures
        if self.params['gradient'] == True:
            data = base_MFS.gradient(data)
        else:
            if self.params['laplacian'] == True:
                print "laplacian!"
                data = base_MFS.laplacian(data)

        xs, ys, zs = data.shape
        print xs, ys, zs

        num_divisions = 8

        #xs_d = xs / num_divisions
        #ys_d = ys / num_divisions
        #zs_d = zs / num_divisions

        dims = 20

        local_mfs = np.zeros((xs, ys, zs, dims))

        #min_diff =  10000.0
        #max_diff = -10000.0

        # window size
        w = 6

        for i in range(xs):
            for j in range(ys):
                for k in range(zs):
                    print i, j, k
                    print max(0, k - w)
                    print min(k + w, zs - 1)
                    mfs = base_MFS.getFDs(
                          data[max(0, i - w) : min(i + w, xs - 1),
                               max(0, j - w): min(j + w, ys - 1),
                               max(0, k - w): min(k + w, zs - 1)])
                    local_mfs[i, j, k] = mfs

                    #d = np.max(mfs) - np.min(mfs)

                    #if d < min_diff:
                    #    min_diff = d

                    #if d > max_diff:
                        #max_diff = d

        return np.mean(local_mfs, axis=(0,1,2))

        #max_fa = np.max(local_mfs)
        #min_fa = np.min(local_mfs)
        #std_fa = np.std(local_mfs)
        #mean_fa = np.mean(local_mfs)

        #return np.array([max_fa, min_fa,
        #                 mean_fa, std_fa,
        #                 max_diff, min_diff])

