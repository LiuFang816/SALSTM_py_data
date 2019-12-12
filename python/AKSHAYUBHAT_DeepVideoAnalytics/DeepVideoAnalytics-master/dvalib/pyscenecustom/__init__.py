#
#         PySceneDetect: Python-Based Video Scene Detector
#   ---------------------------------------------------------------
#     [  Site: http://www.bcastell.com/projects/pyscenedetect/   ]
#     [  Github: https://github.com/Breakthrough/PySceneDetect/  ]
#     [  Documentation: http://pyscenedetect.readthedocs.org/    ]
#
# This file contains all code for the main `scenedetect` module. 
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
import subprocess
import logging

# PySceneDetect Library Imports
import platform
import detectors
import timecodes
import manager
import cli

# Third-Party Library Imports
import cv2
import numpy


# Used for module identification and when printing copyright & version info.
__version__ = 'v0.4'

# About & copyright message string shown for the -v / --version CLI argument.
ABOUT_STRING   = """----------------------------------------------------
PySceneDetect %s
----------------------------------------------------
Site/Updates: https://github.com/Breakthrough/PySceneDetect/
Documentation: http://pyscenedetect.readthedocs.org/

Copyright (C) 2012-2017 Brandon Castellano. All rights reserved.

PySceneDetect is released under the BSD 2-Clause license. See the
included LICENSE file or visit the PySceneDetect website for details.
This software uses the following third-party components:
  > NumPy [Copyright (C) 2005-2016, Numpy Developers]
  > OpenCV [Copyright (C) 2017, Itseez]
  > mkvmerge [Copyright (C) 2005-2016, Matroska]
THE SOFTWARE IS PROVIDED "AS IS" WITHOUT ANY WARRANTY, EXPRESS OR IMPLIED.
""" % __version__


def detect_scenes_file(path, scene_manager):
    """Performs scene detection on passed file using given scene detectors.

    Essentially wraps detect_scenes() while handling all OpenCV interaction.
    For descriptions of arguments and return values that are just passed to
    this function, see the detect_scenes() documentation directly.

    Args:
        path:  A string containing the filename of the video to open.
        scene_manager:  SceneManager interface to scene/detector list and other
            parts of the application state (including user-defined options).

    Returns:
        Tuple containing (video_fps, frames_read, frames_processed), where
        video_fps is a float of the video file's framerate, frames_read is a
        positive, integer number of frames read from the video file, and
        frames_processed is the actual number of frames used.  All values
        are set to -1 if the file could not be opened.
    """

    cap = cv2.VideoCapture()
    frames_read = -1
    frames_processed = -1
    video_fps = -1
    if not scene_manager.timecode_list:
        scene_manager.timecode_list = [0, 0, 0]

    # Attempt to open the passed input (video) file.
    cap.open(path)
    file_name = os.path.split(path)[1]
    if not cap.isOpened():
        if not scene_manager.quiet_mode:
            logging.info('[PySceneDetect] FATAL ERROR - could not open video %s.' % path)
        return (video_fps, frames_read)
    elif not scene_manager.quiet_mode:
        logging.info('[PySceneDetect] Parsing video %s...' % file_name)

    # Print video parameters (resolution, FPS, etc...)
    video_width  = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    video_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    video_fps    = cap.get(cv2.CAP_PROP_FPS)
    if not scene_manager.quiet_mode:
        logging.info('[PySceneDetect] Video Resolution / Framerate: %d x %d / %2.3f FPS' % (
            video_width, video_height, video_fps))
        if scene_manager.downscale_factor >= 2:
            logging.info('[PySceneDetect] Subsampling Enabled (%dx, Resolution = %d x %d)' % (
                scene_manager.downscale_factor,
                video_width / scene_manager.downscale_factor,
                video_height / scene_manager.downscale_factor))
        logging.info('Verify that the above parameters are correct'
              ' (especially framerate, use --force-fps to correct if required).')

    # Convert timecode_list to absolute frames for detect_scenes() function.
    frames_list = []
    for timecode in scene_manager.timecode_list:
        if isinstance(timecode, int):
            frames_list.append(timecode)
        elif isinstance(timecode, float):
            frames_list.append(int(timecode * video_fps))
        elif isinstance(timecode, list) and len(timecode) == 3:
            secs = float(timecode[0] * 60 * 60) + float(timecode[1] * 60) + float(timecode[2])
            frames_list.append(int(secs * video_fps))
        else:
            frames_list.append(0)

    start_frame, end_frame, duration_frames = 0, 0, 0
    if len(frames_list) == 3:
        start_frame, end_frame, duration_frames = (
            frames_list[0], frames_list[1], frames_list[2])

    # Perform scene detection on cap object (modifies scene_list).
    framelist = detect_scenes(cap, scene_manager, start_frame, end_frame, duration_frames)
    # (video_fps, frames_read, frames_processed)
    # Cleanup and return number of frames we read/processed.
    cap.release()
    return framelist


def detect_scenes(cap, scene_manager, start_frame = 0, end_frame = 0, duration_frames = 0):
    """Performs scene detection based on passed video and scene detectors.

    Args:
        cap:  An open cv2.VideoCapture object that is assumed to be at the
            first frame.  Frames are read until cap.read() returns False, and
            the cap object remains open (it can be closed with cap.release()).
        scene_manager:  SceneManager interface to scene/detector list and other
            parts of the application state (including user-defined options).
        image_path_prefix:  Optional.  Filename/path to write images to.
        start_frame:  Optional.  Integer frame number to start processing at.
        end_frame:  Optional.  Integer frame number to stop processing at.
        duration_frames:  Optional.  Integer number of frames to process;
            overrides end_frame if the two values are conflicting.

    Returns:
        Tuple of integers of number of frames read, and number of frames
        actually processed/used for scene detection.
    """
    frames_read = 0
    frames_processed = 0
    frame_metrics = {}
    last_frame = None       # Holds previous frame if needed for save_images.
    framelist = []
    perf_show = True
    perf_last_update_time = time.time()
    perf_last_framecount = 0
    perf_curr_rate = 0
    if scene_manager.perf_update_rate > 0:
        perf_update_rate = float(scene_manager.perf_update_rate)
    else:
        perf_show = False

    # set the end frame if duration_frames is set (overrides end_frame if set)
    if duration_frames > 0:
        end_frame = start_frame + duration_frames

    # If start_frame is set, we drop the required number of frames first.
    # (seeking doesn't work very well, if at all, with OpenCV...)
    while (frames_read < start_frame):
        ret_val = cap.grab()
        frames_read += 1

    stats_file_keys = []
    video_fps = cap.get(cv2.CAP_PROP_FPS)

    while True:
        # If we passed the end point, we stop processing now.
        if end_frame > 0 and frames_read >= end_frame:
            break

        # If frameskip is set, we drop the required number of frames first.
        if scene_manager.frame_skip > 0:
            for _ in range(scene_manager.frame_skip):
                ret_val = cap.grab()
                if not ret_val:
                    break
                frames_read += 1

        (ret_val, im_cap) = cap.read()
        if not ret_val:
            break
        if not frames_read in frame_metrics:
            frame_metrics[frames_read] = dict()
        im_scaled = im_cap
        if scene_manager.downscale_factor > 0:
            im_scaled = im_cap[::scene_manager.downscale_factor,::scene_manager.downscale_factor,:]
        cut_found = False
        for detector in scene_manager.detector_list:
            cut_found = detector.process_frame(frames_read, im_scaled,
                frame_metrics, scene_manager.scene_list) or cut_found
        if scene_manager.stats_writer:
            if not len(stats_file_keys) > 0:
                stats_file_keys = frame_metrics[frames_read].keys()
                if len(stats_file_keys) > 0:
                    scene_manager.stats_writer.writerow(
                        ['Frame Number'] + ['Timecode'] + stats_file_keys)
            if len(stats_file_keys) > 0:
                scene_manager.stats_writer.writerow(
                    [str(frames_read)] +
                    [timecodes.frame_to_timecode(frames_read, video_fps)] +
                    [str(frame_metrics[frames_read][metric]) for metric in stats_file_keys])
        frames_read += 1
        frames_processed += 1
        # periodically show processing speed/performance if requested
        if not scene_manager.quiet_mode and perf_show:
            curr_time = time.time()
            if (curr_time - perf_last_update_time) > perf_update_rate:
                delta_t = curr_time - perf_last_update_time
                delta_f = frames_read - perf_last_framecount
                if delta_t > 0: # and delta_f > 0: # delta_f will always be > 0
                    perf_curr_rate = delta_f / delta_t
                else:
                    perf_curr_rate = 0.0
                perf_last_update_time = curr_time
                perf_last_framecount = frames_read
                logging.info("[PySceneDetect] Current Processing Speed: %3.1f FPS" % perf_curr_rate)
        # save images on scene cuts/breaks if requested (scaled if using -df)
        if cut_found:
            scene_index = len(scene_manager.scene_list)
            output_name = '{}/{}.jpg'.format(scene_manager.save_image_prefix, frames_read)
            framelist.append(frames_read)
            height, width, depth = im_cap.shape
            imgScale = float(scene_manager.rescaled_width) / width
            newX, newY = im_cap.shape[1] * imgScale, im_cap.shape[0] * imgScale
            im_cap = cv2.resize(im_cap, (int(newX), int(newY)))
            cv2.imwrite(output_name, im_cap)

        del last_frame
        last_frame = im_cap.copy()
    # perform any post-processing required by the detectors being used
    for detector in scene_manager.detector_list:
        detector.post_process(scene_manager.scene_list)

    if start_frame > 0:
        frames_read = frames_read - start_frame
    return framelist






def split_input_video(input_path, output_path, timecode_list_str):
    """ Calls the mkvmerge command on the input video, splitting it at the
    passed timecodes, where each scene is written in sequence from 001."""
    #args.output.close()
    logging.info('[PySceneDetect] Splitting video into clips...')
    ret_val = None
    try:
        ret_val = subprocess.call(
            ['mkvmerge',
             '-o', output_path,
             '--split', 'timecodes:%s' % timecode_list_str,
             input_path])
    except ValueError:
        logging.info('[PySceneDetect] Error: mkvmerge could not be found on the system.'
              ' Please install mkvmerge to enable video output support.')
    if ret_val is not None:
        if ret_val != 0:
            logging.info('[PySceneDetect] Error splitting video '
                  '(mkvmerge returned %d).' % ret_val)
        else:
            logging.info('[PySceneDetect] Finished writing scenes to output.')


def output_scene_list(csv_file, smgr, scene_list_tc, scene_start_sec,
                      scene_len_sec):
    ''' Outputs the list of scenes in human-readable format to a CSV file
        for further analysis. '''
    # Output timecodes to CSV file if required (and scenes were found).
    #if args.output and len(smgr.scene_list) > 0:
    if csv_file and len(smgr.scene_list) > 0:
        csv_writer = csv.writer(csv_file) #args.output)
        # Output timecode scene list
        csv_writer.writerow(scene_list_tc)
        # Output detailed, human-readable scene list.
        csv_writer.writerow(["Scene Number", "Frame Number (Start)",
                             "Timecode", "Start Time (seconds)", "Length (seconds)"])
        for i, _ in enumerate(smgr.scene_list):
            csv_writer.writerow([str(i+1), str(smgr.scene_list[i]),
                                 scene_list_tc[i], str(scene_start_sec[i]),
                                 str(scene_len_sec[i])])
