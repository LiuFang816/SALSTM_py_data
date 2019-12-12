# -*- coding: utf-8 -*-
# Copyright 2014-2016 OpenMarket Ltd
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

from ._base import SQLBaseStore
from synapse.util.caches.descriptors import cachedInlineCallbacks, cachedList, cached
from synapse.util.caches.stream_change_cache import StreamChangeCache

from twisted.internet import defer

import logging
import ujson as json


logger = logging.getLogger(__name__)


class ReceiptsStore(SQLBaseStore):
    def __init__(self, hs):
        super(ReceiptsStore, self).__init__(hs)

        self._receipts_stream_cache = StreamChangeCache(
            "ReceiptsRoomChangeCache", self._receipts_id_gen.get_current_token()
        )

    @cachedInlineCallbacks()
    def get_users_with_read_receipts_in_room(self, room_id):
        receipts = yield self.get_receipts_for_room(room_id, "m.read")
        defer.returnValue(set(r['user_id'] for r in receipts))

    def _invalidate_get_users_with_receipts_in_room(self, room_id, receipt_type,
                                                    user_id):
        if receipt_type != "m.read":
            return

        # Returns an ObservableDeferred
        res = self.get_users_with_read_receipts_in_room.cache.get((room_id,), None)

        if res and res.called and user_id in res.result:
            # We'd only be adding to the set, so no point invalidating if the
            # user is already there
            return

        self.get_users_with_read_receipts_in_room.invalidate((room_id,))

    @cached(num_args=2)
    def get_receipts_for_room(self, room_id, receipt_type):
        return self._simple_select_list(
            table="receipts_linearized",
            keyvalues={
                "room_id": room_id,
                "receipt_type": receipt_type,
            },
            retcols=("user_id", "event_id"),
            desc="get_receipts_for_room",
        )

    @cached(num_args=3)
    def get_last_receipt_event_id_for_user(self, user_id, room_id, receipt_type):
        return self._simple_select_one_onecol(
            table="receipts_linearized",
            keyvalues={
                "room_id": room_id,
                "receipt_type": receipt_type,
                "user_id": user_id
            },
            retcol="event_id",
            desc="get_own_receipt_for_user",
            allow_none=True,
        )

    @cachedInlineCallbacks(num_args=2)
    def get_receipts_for_user(self, user_id, receipt_type):
        rows = yield self._simple_select_list(
            table="receipts_linearized",
            keyvalues={
                "user_id": user_id,
                "receipt_type": receipt_type,
            },
            retcols=("room_id", "event_id"),
            desc="get_receipts_for_user",
        )

        defer.returnValue({row["room_id"]: row["event_id"] for row in rows})

    @defer.inlineCallbacks
    def get_receipts_for_user_with_orderings(self, user_id, receipt_type):
        def f(txn):
            sql = (
                "SELECT rl.room_id, rl.event_id,"
                " e.topological_ordering, e.stream_ordering"
                " FROM receipts_linearized AS rl"
                " INNER JOIN events AS e USING (room_id, event_id)"
                " WHERE rl.room_id = e.room_id"
                " AND rl.event_id = e.event_id"
                " AND user_id = ?"
            )
            txn.execute(sql, (user_id,))
            return txn.fetchall()
        rows = yield self.runInteraction(
            "get_receipts_for_user_with_orderings", f
        )
        defer.returnValue({
            row[0]: {
                "event_id": row[1],
                "topological_ordering": row[2],
                "stream_ordering": row[3],
            } for row in rows
        })

    @defer.inlineCallbacks
    def get_linearized_receipts_for_rooms(self, room_ids, to_key, from_key=None):
        """Get receipts for multiple rooms for sending to clients.

        Args:
            room_ids (list): List of room_ids.
            to_key (int): Max stream id to fetch receipts upto.
            from_key (int): Min stream id to fetch receipts from. None fetches
                from the start.

        Returns:
            list: A list of receipts.
        """
        room_ids = set(room_ids)

        if from_key:
            room_ids = yield self._receipts_stream_cache.get_entities_changed(
                room_ids, from_key
            )

        results = yield self._get_linearized_receipts_for_rooms(
            room_ids, to_key, from_key=from_key
        )

        defer.returnValue([ev for res in results.values() for ev in res])

    @cachedInlineCallbacks(num_args=3, tree=True)
    def get_linearized_receipts_for_room(self, room_id, to_key, from_key=None):
        """Get receipts for a single room for sending to clients.

        Args:
            room_ids (str): The room id.
            to_key (int): Max stream id to fetch receipts upto.
            from_key (int): Min stream id to fetch receipts from. None fetches
                from the start.

        Returns:
            list: A list of receipts.
        """
        def f(txn):
            if from_key:
                sql = (
                    "SELECT * FROM receipts_linearized WHERE"
                    " room_id = ? AND stream_id > ? AND stream_id <= ?"
                )

                txn.execute(
                    sql,
                    (room_id, from_key, to_key)
                )
            else:
                sql = (
                    "SELECT * FROM receipts_linearized WHERE"
                    " room_id = ? AND stream_id <= ?"
                )

                txn.execute(
                    sql,
                    (room_id, to_key)
                )

            rows = self.cursor_to_dict(txn)

            return rows

        rows = yield self.runInteraction(
            "get_linearized_receipts_for_room", f
        )

        if not rows:
            defer.returnValue([])

        content = {}
        for row in rows:
            content.setdefault(
                row["event_id"], {}
            ).setdefault(
                row["receipt_type"], {}
            )[row["user_id"]] = json.loads(row["data"])

        defer.returnValue([{
            "type": "m.receipt",
            "room_id": room_id,
            "content": content,
        }])

    @cachedList(cached_method_name="get_linearized_receipts_for_room",
                list_name="room_ids", num_args=3, inlineCallbacks=True)
    def _get_linearized_receipts_for_rooms(self, room_ids, to_key, from_key=None):
        if not room_ids:
            defer.returnValue({})

        def f(txn):
            if from_key:
                sql = (
                    "SELECT * FROM receipts_linearized WHERE"
                    " room_id IN (%s) AND stream_id > ? AND stream_id <= ?"
                ) % (
                    ",".join(["?"] * len(room_ids))
                )
                args = list(room_ids)
                args.extend([from_key, to_key])

                txn.execute(sql, args)
            else:
                sql = (
                    "SELECT * FROM receipts_linearized WHERE"
                    " room_id IN (%s) AND stream_id <= ?"
                ) % (
                    ",".join(["?"] * len(room_ids))
                )

                args = list(room_ids)
                args.append(to_key)

                txn.execute(sql, args)

            return self.cursor_to_dict(txn)

        txn_results = yield self.runInteraction(
            "_get_linearized_receipts_for_rooms", f
        )

        results = {}
        for row in txn_results:
            # We want a single event per room, since we want to batch the
            # receipts by room, event and type.
            room_event = results.setdefault(row["room_id"], {
                "type": "m.receipt",
                "room_id": row["room_id"],
                "content": {},
            })

            # The content is of the form:
            # {"$foo:bar": { "read": { "@user:host": <receipt> }, .. }, .. }
            event_entry = room_event["content"].setdefault(row["event_id"], {})
            receipt_type = event_entry.setdefault(row["receipt_type"], {})

            receipt_type[row["user_id"]] = json.loads(row["data"])

        results = {
            room_id: [results[room_id]] if room_id in results else []
            for room_id in room_ids
        }
        defer.returnValue(results)

    def get_max_receipt_stream_id(self):
        return self._receipts_id_gen.get_current_token()

    def insert_linearized_receipt_txn(self, txn, room_id, receipt_type,
                                      user_id, event_id, data, stream_id):
        txn.call_after(
            self.get_receipts_for_room.invalidate, (room_id, receipt_type)
        )
        txn.call_after(
            self._invalidate_get_users_with_receipts_in_room,
            room_id, receipt_type, user_id,
        )
        txn.call_after(
            self.get_receipts_for_user.invalidate, (user_id, receipt_type)
        )
        # FIXME: This shouldn't invalidate the whole cache
        txn.call_after(self.get_linearized_receipts_for_room.invalidate_many, (room_id,))

        txn.call_after(
            self._receipts_stream_cache.entity_has_changed,
            room_id, stream_id
        )

        txn.call_after(
            self.get_last_receipt_event_id_for_user.invalidate,
            (user_id, room_id, receipt_type)
        )

        res = self._simple_select_one_txn(
            txn,
            table="events",
            retcols=["topological_ordering", "stream_ordering"],
            keyvalues={"event_id": event_id},
            allow_none=True
        )

        topological_ordering = int(res["topological_ordering"]) if res else None
        stream_ordering = int(res["stream_ordering"]) if res else None

        # We don't want to clobber receipts for more recent events, so we
        # have to compare orderings of existing receipts
        sql = (
            "SELECT topological_ordering, stream_ordering, event_id FROM events"
            " INNER JOIN receipts_linearized as r USING (event_id, room_id)"
            " WHERE r.room_id = ? AND r.receipt_type = ? AND r.user_id = ?"
        )

        txn.execute(sql, (room_id, receipt_type, user_id))
        results = txn.fetchall()

        if results and topological_ordering:
            for to, so, _ in results:
                if int(to) > topological_ordering:
                    return False
                elif int(to) == topological_ordering and int(so) >= stream_ordering:
                    return False

        self._simple_delete_txn(
            txn,
            table="receipts_linearized",
            keyvalues={
                "room_id": room_id,
                "receipt_type": receipt_type,
                "user_id": user_id,
            }
        )

        self._simple_insert_txn(
            txn,
            table="receipts_linearized",
            values={
                "stream_id": stream_id,
                "room_id": room_id,
                "receipt_type": receipt_type,
                "user_id": user_id,
                "event_id": event_id,
                "data": json.dumps(data),
            }
        )

        if receipt_type == "m.read" and topological_ordering:
            self._remove_old_push_actions_before_txn(
                txn,
                room_id=room_id,
                user_id=user_id,
                topological_ordering=topological_ordering,
            )

        return True

    @defer.inlineCallbacks
    def insert_receipt(self, room_id, receipt_type, user_id, event_ids, data):
        """Insert a receipt, either from local client or remote server.

        Automatically does conversion between linearized and graph
        representations.
        """
        if not event_ids:
            return

        if len(event_ids) == 1:
            linearized_event_id = event_ids[0]
        else:
            # we need to points in graph -> linearized form.
            # TODO: Make this better.
            def graph_to_linear(txn):
                query = (
                    "SELECT event_id WHERE room_id = ? AND stream_ordering IN ("
                    " SELECT max(stream_ordering) WHERE event_id IN (%s)"
                    ")"
                ) % (",".join(["?"] * len(event_ids)))

                txn.execute(query, [room_id] + event_ids)
                rows = txn.fetchall()
                if rows:
                    return rows[0][0]
                else:
                    raise RuntimeError("Unrecognized event_ids: %r" % (event_ids,))

            linearized_event_id = yield self.runInteraction(
                "insert_receipt_conv", graph_to_linear
            )

        stream_id_manager = self._receipts_id_gen.get_next()
        with stream_id_manager as stream_id:
            have_persisted = yield self.runInteraction(
                "insert_linearized_receipt",
                self.insert_linearized_receipt_txn,
                room_id, receipt_type, user_id, linearized_event_id,
                data,
                stream_id=stream_id,
            )

            if not have_persisted:
                defer.returnValue(None)

        yield self.insert_graph_receipt(
            room_id, receipt_type, user_id, event_ids, data
        )

        max_persisted_id = self._receipts_id_gen.get_current_token()

        defer.returnValue((stream_id, max_persisted_id))

    def insert_graph_receipt(self, room_id, receipt_type, user_id, event_ids,
                             data):
        return self.runInteraction(
            "insert_graph_receipt",
            self.insert_graph_receipt_txn,
            room_id, receipt_type, user_id, event_ids, data
        )

    def insert_graph_receipt_txn(self, txn, room_id, receipt_type,
                                 user_id, event_ids, data):
        txn.call_after(
            self.get_receipts_for_room.invalidate, (room_id, receipt_type)
        )
        txn.call_after(
            self._invalidate_get_users_with_receipts_in_room,
            room_id, receipt_type, user_id,
        )
        txn.call_after(
            self.get_receipts_for_user.invalidate, (user_id, receipt_type)
        )
        # FIXME: This shouldn't invalidate the whole cache
        txn.call_after(self.get_linearized_receipts_for_room.invalidate_many, (room_id,))

        self._simple_delete_txn(
            txn,
            table="receipts_graph",
            keyvalues={
                "room_id": room_id,
                "receipt_type": receipt_type,
                "user_id": user_id,
            }
        )
        self._simple_insert_txn(
            txn,
            table="receipts_graph",
            values={
                "room_id": room_id,
                "receipt_type": receipt_type,
                "user_id": user_id,
                "event_ids": json.dumps(event_ids),
                "data": json.dumps(data),
            }
        )

    def get_all_updated_receipts(self, last_id, current_id, limit=None):
        if last_id == current_id:
            return defer.succeed([])

        def get_all_updated_receipts_txn(txn):
            sql = (
                "SELECT stream_id, room_id, receipt_type, user_id, event_id, data"
                " FROM receipts_linearized"
                " WHERE ? < stream_id AND stream_id <= ?"
                " ORDER BY stream_id ASC"
            )
            args = [last_id, current_id]
            if limit is not None:
                sql += " LIMIT ?"
                args.append(limit)
            txn.execute(sql, args)

            return txn.fetchall()
        return self.runInteraction(
            "get_all_updated_receipts", get_all_updated_receipts_txn
        )
