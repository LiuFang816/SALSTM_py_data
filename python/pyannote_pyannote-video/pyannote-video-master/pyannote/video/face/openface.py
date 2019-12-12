# Copyright 2015 Carnegie Mellon University
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

import atexit
import binascii
from subprocess import Popen, PIPE
import os
import os.path
import sys

import numpy as np
import cv2


class TorchWrap(object):
    # Warning: This is very unstable!
    # Please join us in improving it at:
    #   https://github.com/cmusatyalab/openface/issues/1
    #   https://github.com/cmusatyalab/openface/issues/4

    def __init__(self, torch='th',
                 model='nn4.v2.t7', size=96, cuda=False):

        super(TorchWrap, self).__init__()

        luaFile = os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            'openface_server.lua')

        self.cmd = [torch, luaFile, '-model', model, '-imgDim', str(size)]

        if cuda:
            self.cmd.append('-cuda')

        self.p = Popen(self.cmd, stdin=PIPE, stdout=PIPE,
                       stderr=PIPE, bufsize=0)

        def exitHandler():
            if self.p.poll() is None:
                self.p.kill()

        atexit.register(exitHandler)

    def forwardPath(self, imgPath):
        rc = self.p.poll()
        if rc is not None and rc != 0:
            raise Exception("""


OpenFace: `openface_server.lua` subprocess has died.
Is the Torch command `th` on your PATH? Check with `which th`.

Diagnostic information:

cmd: {}

============

stdout: {}

============

stderr: {}
""".format(self.cmd, self.p.stdout.read(), self.p.stderr.read()))

        self.p.stdin.write(imgPath + "\n")
        output = self.p.stdout.readline()
        try:
            return [float(x) for x in output.strip().split(',')]
        except Exception as e:
            self.p.kill()
            stdout, stderr = self.p.communicate()
            print("""


Error getting result from Torch subprocess.

Line read: {}

Exception:

{}

============

stdout: {}

============

stderr: {}
""".format(output, str(e), stdout, stderr))
            sys.exit(-1)

    def forwardImage(self, bgr):
        t = '/tmp/openface-torchwrap-{}.png'.format(
            binascii.b2a_hex(os.urandom(8)))
        cv2.imwrite(t, bgr)
        rep = np.array(self.forwardPath(t))
        os.remove(t)
        return rep
