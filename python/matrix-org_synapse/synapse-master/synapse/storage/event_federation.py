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

from twisted.internet import defer

from ._base import SQLBaseStore
from synapse.api.errors import StoreError
from synapse.util.caches.descriptors import cached
from unpaddedbase64 import encode_base64

import logging
from Queue import PriorityQueue, Empty


logger = logging.getLogger(__name__)


class EventFederationStore(SQLBaseStore):
    """ Responsible for storing and serving up the various graphs associated
    with an event. Including the main event graph and the auth chains for an
    event.

    Also has methods for getting the front (latest) and back (oldest) edges
    of the event graphs. These are used to generate the parents for new events
    and backfilling from another server respectively.
    """

    def __init__(self, hs):
        super(EventFederationStore, self).__init__(hs)

        hs.get_clock().looping_call(
            self._delete_old_forward_extrem_cache, 60 * 60 * 1000
        )

    def get_auth_chain(self, event_ids):
        return self.get_auth_chain_ids(event_ids).addCallback(self._get_events)

    def get_auth_chain_ids(self, event_ids):
        return self.runInteraction(
            "get_auth_chain_ids",
            self._get_auth_chain_ids_txn,
            event_ids
        )

    def _get_auth_chain_ids_txn(self, txn, event_ids):
        results = set()

        base_sql = (
            "SELECT auth_id FROM event_auth WHERE event_id IN (%s)"
        )

        front = set(event_ids)
        while front:
            new_front = set()
            front_list = list(front)
            chunks = [
                front_list[x:x + 100]
                for x in xrange(0, len(front), 100)
            ]
            for chunk in chunks:
                txn.execute(
                    base_sql % (",".join(["?"] * len(chunk)),),
                    chunk
                )
                new_front.update([r[0] for r in txn.fetchall()])

            new_front -= results

            front = new_front
            results.update(front)

        return list(results)

    def get_oldest_events_in_room(self, room_id):
        return self.runInteraction(
            "get_oldest_events_in_room",
            self._get_oldest_events_in_room_txn,
            room_id,
        )

    def get_oldest_events_with_depth_in_room(self, room_id):
        return self.runInteraction(
            "get_oldest_events_with_depth_in_room",
            self.get_oldest_events_with_depth_in_room_txn,
            room_id,
        )

    def get_oldest_events_with_depth_in_room_txn(self, txn, room_id):
        sql = (
            "SELECT b.event_id, MAX(e.depth) FROM events as e"
            " INNER JOIN event_edges as g"
            " ON g.event_id = e.event_id AND g.room_id = e.room_id"
            " INNER JOIN event_backward_extremities as b"
            " ON g.prev_event_id = b.event_id AND g.room_id = b.room_id"
            " WHERE b.room_id = ? AND g.is_state is ?"
            " GROUP BY b.event_id"
        )

        txn.execute(sql, (room_id, False,))

        return dict(txn.fetchall())

    def _get_oldest_events_in_room_txn(self, txn, room_id):
        return self._simple_select_onecol_txn(
            txn,
            table="event_backward_extremities",
            keyvalues={
                "room_id": room_id,
            },
            retcol="event_id",
        )

    def get_latest_event_ids_and_hashes_in_room(self, room_id):
        return self.runInteraction(
            "get_latest_event_ids_and_hashes_in_room",
            self._get_latest_event_ids_and_hashes_in_room,
            room_id,
        )

    @cached(max_entries=5000, iterable=True)
    def get_latest_event_ids_in_room(self, room_id):
        return self._simple_select_onecol(
            table="event_forward_extremities",
            keyvalues={
                "room_id": room_id,
            },
            retcol="event_id",
            desc="get_latest_event_ids_in_room",
        )

    def _get_latest_event_ids_and_hashes_in_room(self, txn, room_id):
        sql = (
            "SELECT e.event_id, e.depth FROM events as e "
            "INNER JOIN event_forward_extremities as f "
            "ON e.event_id = f.event_id "
            "AND e.room_id = f.room_id "
            "WHERE f.room_id = ?"
        )

        txn.execute(sql, (room_id, ))

        results = []
        for event_id, depth in txn.fetchall():
            hashes = self._get_event_reference_hashes_txn(txn, event_id)
            prev_hashes = {
                k: encode_base64(v) for k, v in hashes.items()
                if k == "sha256"
            }
            results.append((event_id, prev_hashes, depth))

        return results

    def get_min_depth(self, room_id):
        """ For hte given room, get the minimum depth we have seen for it.
        """
        return self.runInteraction(
            "get_min_depth",
            self._get_min_depth_interaction,
            room_id,
        )

    @defer.inlineCallbacks
    def get_max_depth_of_events(self, event_ids):
        sql = (
            "SELECT MAX(depth) FROM events WHERE event_id IN (%s)"
        ) % (",".join(["?"] * len(event_ids)),)

        rows = yield self._execute(
            "get_max_depth_of_events", None,
            sql, *event_ids
        )

        if rows:
            defer.returnValue(rows[0][0])
        else:
            defer.returnValue(1)

    def _get_min_depth_interaction(self, txn, room_id):
        min_depth = self._simple_select_one_onecol_txn(
            txn,
            table="room_depth",
            keyvalues={"room_id": room_id},
            retcol="min_depth",
            allow_none=True,
        )

        return int(min_depth) if min_depth is not None else None

    def _update_min_depth_for_room_txn(self, txn, room_id, depth):
        min_depth = self._get_min_depth_interaction(txn, room_id)

        do_insert = depth < min_depth if min_depth else True

        if do_insert:
            self._simple_upsert_txn(
                txn,
                table="room_depth",
                keyvalues={
                    "room_id": room_id,
                },
                values={
                    "min_depth": depth,
                },
            )

    def _handle_mult_prev_events(self, txn, events):
        """
        For the given event, update the event edges table and forward and
        backward extremities tables.
        """
        self._simple_insert_many_txn(
            txn,
            table="event_edges",
            values=[
                {
                    "event_id": ev.event_id,
                    "prev_event_id": e_id,
                    "room_id": ev.room_id,
                    "is_state": False,
                }
                for ev in events
                for e_id, _ in ev.prev_events
            ],
        )

        self._update_backward_extremeties(txn, events)

    def _update_backward_extremeties(self, txn, events):
        """Updates the event_backward_extremities tables based on the new/updated
        events being persisted.

        This is called for new events *and* for events that were outliers, but
        are now being persisted as non-outliers.

        Forward extremities are handled when we first start persisting the events.
        """
        events_by_room = {}
        for ev in events:
            events_by_room.setdefault(ev.room_id, []).append(ev)

        query = (
            "INSERT INTO event_backward_extremities (event_id, room_id)"
            " SELECT ?, ? WHERE NOT EXISTS ("
            " SELECT 1 FROM event_backward_extremities"
            " WHERE event_id = ? AND room_id = ?"
            " )"
            " AND NOT EXISTS ("
            " SELECT 1 FROM events WHERE event_id = ? AND room_id = ? "
            " AND outlier = ?"
            " )"
        )

        txn.executemany(query, [
            (e_id, ev.room_id, e_id, ev.room_id, e_id, ev.room_id, False)
            for ev in events for e_id, _ in ev.prev_events
            if not ev.internal_metadata.is_outlier()
        ])

        query = (
            "DELETE FROM event_backward_extremities"
            " WHERE event_id = ? AND room_id = ?"
        )
        txn.executemany(
            query,
            [
                (ev.event_id, ev.room_id) for ev in events
                if not ev.internal_metadata.is_outlier()
            ]
        )

    def get_forward_extremeties_for_room(self, room_id, stream_ordering):
        # We want to make the cache more effective, so we clamp to the last
        # change before the given ordering.
        last_change = self._events_stream_cache.get_max_pos_of_last_change(room_id)

        # We don't always have a full stream_to_exterm_id table, e.g. after
        # the upgrade that introduced it, so we make sure we never ask for a
        # try and pin to a stream_ordering from before a restart
        last_change = max(self._stream_order_on_start, last_change)

        if last_change > self.stream_ordering_month_ago:
            stream_ordering = min(last_change, stream_ordering)

        return self._get_forward_extremeties_for_room(room_id, stream_ordering)

    @cached(max_entries=5000, num_args=2)
    def _get_forward_extremeties_for_room(self, room_id, stream_ordering):
        """For a given room_id and stream_ordering, return the forward
        extremeties of the room at that point in "time".

        Throws a StoreError if we have since purged the index for
        stream_orderings from that point.
        """

        if stream_ordering <= self.stream_ordering_month_ago:
            raise StoreError(400, "stream_ordering too old")

        sql = ("""
                SELECT event_id FROM stream_ordering_to_exterm
                INNER JOIN (
                    SELECT room_id, MAX(stream_ordering) AS stream_ordering
                    FROM stream_ordering_to_exterm
                    WHERE stream_ordering <= ? GROUP BY room_id
                ) AS rms USING (room_id, stream_ordering)
                WHERE room_id = ?
        """)

        def get_forward_extremeties_for_room_txn(txn):
            txn.execute(sql, (stream_ordering, room_id))
            rows = txn.fetchall()
            return [event_id for event_id, in rows]

        return self.runInteraction(
            "get_forward_extremeties_for_room",
            get_forward_extremeties_for_room_txn
        )

    def _delete_old_forward_extrem_cache(self):
        def _delete_old_forward_extrem_cache_txn(txn):
            # Delete entries older than a month, while making sure we don't delete
            # the only entries for a room.
            sql = ("""
                DELETE FROM stream_ordering_to_exterm
                WHERE
                room_id IN (
                    SELECT room_id
                    FROM stream_ordering_to_exterm
                    WHERE stream_ordering > ?
                ) AND stream_ordering < ?
            """)
            txn.execute(
                sql,
                (self.stream_ordering_month_ago, self.stream_ordering_month_ago,)
            )
        return self.runInteraction(
            "_delete_old_forward_extrem_cache",
            _delete_old_forward_extrem_cache_txn
        )

    def get_backfill_events(self, room_id, event_list, limit):
        """Get a list of Events for a given topic that occurred before (and
        including) the events in event_list. Return a list of max size `limit`

        Args:
            txn
            room_id (str)
            event_list (list)
            limit (int)
        """
        return self.runInteraction(
            "get_backfill_events",
            self._get_backfill_events, room_id, event_list, limit
        ).addCallback(
            self._get_events
        ).addCallback(
            lambda l: sorted(l, key=lambda e: -e.depth)
        )

    def _get_backfill_events(self, txn, room_id, event_list, limit):
        logger.debug(
            "_get_backfill_events: %s, %s, %s",
            room_id, repr(event_list), limit
        )

        event_results = set()

        # We want to make sure that we do a breadth-first, "depth" ordered
        # search.

        query = (
            "SELECT depth, prev_event_id FROM event_edges"
            " INNER JOIN events"
            " ON prev_event_id = events.event_id"
            " AND event_edges.room_id = events.room_id"
            " WHERE event_edges.room_id = ? AND event_edges.event_id = ?"
            " AND event_edges.is_state = ?"
            " LIMIT ?"
        )

        queue = PriorityQueue()

        for event_id in event_list:
            depth = self._simple_select_one_onecol_txn(
                txn,
                table="events",
                keyvalues={
                    "event_id": event_id,
                },
                retcol="depth",
                allow_none=True,
            )

            if depth:
                queue.put((-depth, event_id))

        while not queue.empty() and len(event_results) < limit:
            try:
                _, event_id = queue.get_nowait()
            except Empty:
                break

            if event_id in event_results:
                continue

            event_results.add(event_id)

            txn.execute(
                query,
                (room_id, event_id, False, limit - len(event_results))
            )

            for row in txn.fetchall():
                if row[1] not in event_results:
                    queue.put((-row[0], row[1]))

        return event_results

    @defer.inlineCallbacks
    def get_missing_events(self, room_id, earliest_events, latest_events,
                           limit, min_depth):
        ids = yield self.runInteraction(
            "get_missing_events",
            self._get_missing_events,
            room_id, earliest_events, latest_events, limit, min_depth
        )

        events = yield self._get_events(ids)

        events = sorted(
            [ev for ev in events if ev.depth >= min_depth],
            key=lambda e: e.depth,
        )

        defer.returnValue(events[:limit])

    def _get_missing_events(self, txn, room_id, earliest_events, latest_events,
                            limit, min_depth):

        earliest_events = set(earliest_events)
        front = set(latest_events) - earliest_events

        event_results = set()

        query = (
            "SELECT prev_event_id FROM event_edges "
            "WHERE room_id = ? AND event_id = ? AND is_state = ? "
            "LIMIT ?"
        )

        while front and len(event_results) < limit:
            new_front = set()
            for event_id in front:
                txn.execute(
                    query,
                    (room_id, event_id, False, limit - len(event_results))
                )

                for e_id, in txn.fetchall():
                    new_front.add(e_id)

            new_front -= earliest_events
            new_front -= event_results

            front = new_front
            event_results |= new_front

        return event_results

    def clean_room_for_join(self, room_id):
        return self.runInteraction(
            "clean_room_for_join",
            self._clean_room_for_join_txn,
            room_id,
        )

    def _clean_room_for_join_txn(self, txn, room_id):
        query = "DELETE FROM event_forward_extremities WHERE room_id = ?"

        txn.execute(query, (room_id,))
        txn.call_after(self.get_latest_event_ids_in_room.invalidate, (room_id,))
