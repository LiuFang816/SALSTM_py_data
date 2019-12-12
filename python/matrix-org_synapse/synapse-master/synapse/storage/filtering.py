# -*- coding: utf-8 -*-
# Copyright 2015, 2016 OpenMarket Ltd
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

from twisted.internet import defer

from ._base import SQLBaseStore
from synapse.api.errors import SynapseError, Codes
from synapse.util.caches.descriptors import cachedInlineCallbacks

import simplejson as json


class FilteringStore(SQLBaseStore):
    @cachedInlineCallbacks(num_args=2)
    def get_user_filter(self, user_localpart, filter_id):
        # filter_id is BIGINT UNSIGNED, so if it isn't a number, fail
        # with a coherent error message rather than 500 M_UNKNOWN.
        try:
            int(filter_id)
        except ValueError:
            raise SynapseError(400, "Invalid filter ID", Codes.INVALID_PARAM)

        def_json = yield self._simple_select_one_onecol(
            table="user_filters",
            keyvalues={
                "user_id": user_localpart,
                "filter_id": filter_id,
            },
            retcol="filter_json",
            allow_none=False,
            desc="get_user_filter",
        )

        defer.returnValue(json.loads(str(def_json).decode("utf-8")))

    def add_user_filter(self, user_localpart, user_filter):
        def_json = json.dumps(user_filter).encode("utf-8")

        # Need an atomic transaction to SELECT the maximal ID so far then
        # INSERT a new one
        def _do_txn(txn):
            sql = (
                "SELECT MAX(filter_id) FROM user_filters "
                "WHERE user_id = ?"
            )
            txn.execute(sql, (user_localpart,))
            max_id = txn.fetchone()[0]
            if max_id is None:
                filter_id = 0
            else:
                filter_id = max_id + 1

            sql = (
                "INSERT INTO user_filters (user_id, filter_id, filter_json)"
                "VALUES(?, ?, ?)"
            )
            txn.execute(sql, (user_localpart, filter_id, def_json))

            return filter_id

        return self.runInteraction("add_user_filter", _do_txn)
