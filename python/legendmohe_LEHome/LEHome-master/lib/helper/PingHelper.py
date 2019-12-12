#!/usr/bin/env python
# encoding: utf-8

# Copyright 2014 Xinyu, He <legendmohe@foxmail.com>
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#   http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import threading
import json
import time
import zmq
from util.Res import Res
from util.log import *

class PingHelper(object):

    def __init__(self, server_ip, settings):
        self._server_ip = server_ip
        self._devices = settings["device"]

    def device_ip_for_name(self, name):
        return self._devices.get(name)

    def online(self, device_ip):
        rep = self._send_request(device_ip)
        if rep is None:
            return None
        res = json.loads(rep)["res"]
        if res == "error":
            INFO('ping server error.')
            return None
        return res['online']

    def _send_request(self, cmd):
        DEBUG("send tag request to %s for %s" % (self._server_ip, cmd))
        context = zmq.Context()
        socket = context.socket(zmq.REQ)
        socket.setsockopt(zmq.LINGER, 0)
        socket.connect(self._server_ip)
        socket.send_string(cmd)
        rep = None
        try:
            poller = zmq.Poller()
            poller.register(socket, zmq.POLLIN)
            if poller.poll(5*1000):
                rep = socket.recv_string()
                DEBUG("recv msgs:" + rep)
        except:
            WARN("socket timeout.")
        socket.close()
        return rep
