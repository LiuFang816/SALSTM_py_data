#!/usr/bin/env python2
#
# Copyright 2015-2016 Carnegie Mellon University
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
#
# This script downloads the VGG Face Dataset from
# http://www.robots.ox.ac.uk/~vgg/data/vgg_face/
# and aligns the images with OpenFace.
#
# Brandon Amos
# 2016-02-29

import argparse
import cv2
import dlib
import os
import urllib2
import hashlib

from multiprocessing import Pool

import openface
from openface.helper import mkdirP

fileDir = os.path.dirname(os.path.realpath(__file__))
modelDir = os.path.join(fileDir, '..', '..', 'models')
dlibModelDir = os.path.join(modelDir, 'dlib')
openfaceModelDir = os.path.join(modelDir, 'openface')

landmarkIndices = openface.AlignDlib.OUTER_EYES_AND_NOSE

parser = argparse.ArgumentParser()
parser.add_argument('--dlibFacePredictor', type=str, help="Path to dlib's face predictor.",
                    default=os.path.join(dlibModelDir, "shape_predictor_68_face_landmarks.dat"))
parser.add_argument("--txt", help="VGG's directory of text files of people with images.",
                    default='raw-txt')
parser.add_argument("--raw", help="Directory to save raw images to.",
                    default='raw')
parser.add_argument("--aligned", help="Directory to save aligned images to.",
                    default='aligned')

args = parser.parse_args()

align = openface.AlignDlib(args.dlibFacePredictor)

jobs = []
for person in os.listdir(args.txt):
    fullPersonPath = os.path.join(args.txt, person)
    with open(fullPersonPath, 'r') as f:
        contents = f.readlines()

    for line in contents:
        id, uid, url, l, t, r, b, pose, detection, curation = line.split()
        l, t, r, b = [int(float(x)) for x in [l, t, r, b]]
        # if int(curation) == 1:
        jobs.append((person[:-4], url, (l, t, r, b)))


def download(person, url, bb):
    imgName = os.path.basename(url)
    rawPersonPath = os.path.join(args.raw, person)
    rawImgPath = os.path.join(rawPersonPath, imgName)
    alignedPersonPath = os.path.join(args.aligned, person)
    alignedImgPath = os.path.join(alignedPersonPath,
                                  hashlib.md5(imgName).hexdigest() + ".png")

    mkdirP(rawPersonPath)
    mkdirP(alignedPersonPath)

    if not os.path.isfile(rawImgPath):
        urlF = urllib2.urlopen(url, timeout=5)
        with open(rawImgPath, 'wb') as f:
            f.write(urlF.read())

    if not os.path.isfile(alignedImgPath):
        bgr = cv2.imread(rawImgPath)
        if bgr is None:
            return

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        dlibBB = dlib.rectangle(*bb)
        outRgb = align.align(96, rgb,
                             bb=dlibBB,
                             landmarkIndices=landmarkIndices)

        if outRgb is not None:
            outBgr = cv2.cvtColor(outRgb, cv2.COLOR_RGB2BGR)
            cv2.imwrite(alignedImgPath, outBgr)


def download_packed(args):
    try:
        download(*args)
    except Exception as e:
        print("\n".join((str(args), str(e))))
        pass

pool = Pool(16)
pool.map(download_packed, jobs)
