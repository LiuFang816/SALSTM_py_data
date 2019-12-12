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

from ._base import BaseSlavedStore
from synapse.storage import DataStore
from synapse.storage.room import RoomStore
from ._slaved_id_tracker import SlavedIdTracker


class RoomStore(BaseSlavedStore):
    def __init__(self, db_conn, hs):
        super(RoomStore, self).__init__(db_conn, hs)
        self._public_room_id_gen = SlavedIdTracker(
            db_conn, "public_room_list_stream", "stream_id"
        )

    get_public_room_ids = DataStore.get_public_room_ids.__func__
    get_current_public_room_stream_id = (
        DataStore.get_current_public_room_stream_id.__func__
    )
    get_public_room_ids_at_stream_id = (
        RoomStore.__dict__["get_public_room_ids_at_stream_id"]
    )
    get_public_room_ids_at_stream_id_txn = (
        DataStore.get_public_room_ids_at_stream_id_txn.__func__
    )
    get_published_at_stream_id_txn = (
        DataStore.get_published_at_stream_id_txn.__func__
    )
    get_public_room_changes = DataStore.get_public_room_changes.__func__

    def stream_positions(self):
        result = super(RoomStore, self).stream_positions()
        result["public_rooms"] = self._public_room_id_gen.get_current_token()
        return result

    def process_replication(self, result):
        stream = result.get("public_rooms")
        if stream:
            self._public_room_id_gen.advance(int(stream["position"]))

        return super(RoomStore, self).process_replication(result)
