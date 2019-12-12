"""
Copyright (c) 2013 Rodrigo Baravalle
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
from imfractal import *

import Image
import time
import matplotlib.pyplot as plt
from pylab import *

import os

cantMeasures = 14 # different ways for obtaining identical measures

def createFolders(dirs):

    for f in dirs:
        if not os.path.isdir(f): 
            os.mkdir (f) 

def readBVTV(filename):

    res = np.zeros(27,cantMeasures)

    i = 0
    with open(filename, "r") as ins:
        for line in ins:
            if(i != 0 and i != 1):
                arr = line.split()
                print arr
                print len(arr)
                exit()
                #print res.shape, len(arr), i
                #res[i-2] = arr[4]
                # ?? VER
                res[i-2,0] = arr[3]
                res[i-2,1] = arr[5]
                res[i-2,2] = arr[7]
                res[i-2,3] = arr[8]
                res[i-2,4] = arr[5]
                res[i-2,5] = arr[7]
                res[i-2,6] = arr[8]
                res[i-2,7] = arr[5]
                res[i-2,8] = arr[7]
                res[i-2,9] = arr[8]
                res[i-2,10] = arr[5]
                res[i-2,11] = arr[7]
                res[i-2,12] = arr[8]
                res[i-2,13] = arr[5]


            i = i+1
            if(i>=29): break

    print res
    return res


def do_test():

    path = "/home/rodrigo/rodrigo/rodrigo/boneMeasures/SiCopyRodrigo/Win32Release/DSM-RLM/"

    patients = ["5c", "6b", "8b", "8c", "V12"]
    scans = ["01", "02", "03", "M1", "M2", "XCT"]
    # amount of volumes of interest
    vois = 27


    # bone's multifractal spectra database
    bvtvs = np.zeros([len(patients),len(scans),vois,cantMeasures])

    ii = 0
    for i in patients:
        jj = 0
        for j in scans:

            filename = path+i+j+".txt"

            print i,j
            print filename

            measures[ii,jj] = readMeasures(filename)

            jj = jj+1

        ii = ii+1


    np.save("measures",measures)

    
