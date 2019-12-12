# -*- coding: utf-8 -*-
# Copyright 2016 OpenMarket Ltd
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


from tests import unittest

from twisted.internet import defer

from synapse.util.async import Linearizer


class LinearizerTestCase(unittest.TestCase):

    @defer.inlineCallbacks
    def test_linearizer(self):
        linearizer = Linearizer()

        key = object()

        d1 = linearizer.queue(key)
        cm1 = yield d1

        d2 = linearizer.queue(key)
        self.assertFalse(d2.called)

        with cm1:
            self.assertFalse(d2.called)

        self.assertTrue(d2.called)

        with (yield d2):
            pass
