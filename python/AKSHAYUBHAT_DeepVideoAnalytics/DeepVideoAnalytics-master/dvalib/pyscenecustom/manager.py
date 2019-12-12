#
#         PySceneDetect: Python-Based Video Scene Detector
#   ---------------------------------------------------------------
#     [  Site: http://www.bcastell.com/projects/pyscenedetect/   ]
#     [  Github: https://github.com/Breakthrough/PySceneDetect/  ]
#     [  Documentation: http://pyscenedetect.readthedocs.org/    ]
#
# This file contains the SceneManager class, which provides a
# consistent interface to the application state, including the current
# scene list, user-defined options, and any shared objects.
#
# Copyright (C) 2012-2017 Brandon Castellano <http://www.bcastell.com>.
#
# PySceneDetect is licensed under the BSD 2-Clause License; see the
# included LICENSE file or visit one of the following pages for details:
#  - http://www.bcastell.com/projects/pyscenedetect/
#  - https://github.com/Breakthrough/PySceneDetect/
#
# This software uses Numpy and OpenCV; see the LICENSE-NUMPY and
# LICENSE-OPENCV files or visit one of above URLs for details.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# IN NO EVENT SHALL THE AUTHORS BE LIABLE FOR ANY CLAIM, DAMAGES OR
# OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
# ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
# OTHER DEALINGS IN THE SOFTWARE.
#

# Standard Library Imports
from __future__ import print_function
import sys
import os
import argparse
import time
import csv

# PySceneDetect Library Imports
import detectors

# Third-Party Library Imports
import cv2
import numpy

# detectors = [scenedetect.detectors.ThresholdDetector(threshold = 16, min_percent = 0.9)]


class SceneManager(object):

    def __init__(self, save_image_prefix, rescaled_width):
        self.scene_list = list()
        args = {}
        self.rescaled_width = rescaled_width
        self.args = args
        self.detector = None
        self.detection_method = 'content'
        self.scene_detectors= detectors.get_available()
        self.detector = self.scene_detectors['content'](30.0, 20)
        self.detector_list = [ self.detector ]
        self.downscale_factor = 4
        self.frame_skip = 10
        self.save_images = True
        self.save_image_prefix = save_image_prefix
        self.timecode_list = [None,None,None]
        self.quiet_mode = False
        self.perf_update_rate = -1
        self.stats_writer = None
        # if args['stats_file']:
        #     self.stats_writer = csv.writer(args['stats_file'])
        self.cap = None
            





