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
from twisted.internet import defer

from canonicaljson import encode_canonical_json

from synapse.util.caches.descriptors import cachedInlineCallbacks, cachedList

import logging
import simplejson as json
import types

logger = logging.getLogger(__name__)


class PusherStore(SQLBaseStore):
    def _decode_pushers_rows(self, rows):
        for r in rows:
            dataJson = r['data']
            r['data'] = None
            try:
                if isinstance(dataJson, types.BufferType):
                    dataJson = str(dataJson).decode("UTF8")

                r['data'] = json.loads(dataJson)
            except Exception as e:
                logger.warn(
                    "Invalid JSON in data for pusher %d: %s, %s",
                    r['id'], dataJson, e.message,
                )
                pass

            if isinstance(r['pushkey'], types.BufferType):
                r['pushkey'] = str(r['pushkey']).decode("UTF8")

        return rows

    @defer.inlineCallbacks
    def user_has_pusher(self, user_id):
        ret = yield self._simple_select_one_onecol(
            "pushers", {"user_name": user_id}, "id", allow_none=True
        )
        defer.returnValue(ret is not None)

    def get_pushers_by_app_id_and_pushkey(self, app_id, pushkey):
        return self.get_pushers_by({
            "app_id": app_id,
            "pushkey": pushkey,
        })

    def get_pushers_by_user_id(self, user_id):
        return self.get_pushers_by({
            "user_name": user_id,
        })

    @defer.inlineCallbacks
    def get_pushers_by(self, keyvalues):
        ret = yield self._simple_select_list(
            "pushers", keyvalues,
            [
                "id",
                "user_name",
                "access_token",
                "profile_tag",
                "kind",
                "app_id",
                "app_display_name",
                "device_display_name",
                "pushkey",
                "ts",
                "lang",
                "data",
                "last_stream_ordering",
                "last_success",
                "failing_since",
            ], desc="get_pushers_by"
        )
        defer.returnValue(self._decode_pushers_rows(ret))

    @defer.inlineCallbacks
    def get_all_pushers(self):
        def get_pushers(txn):
            txn.execute("SELECT * FROM pushers")
            rows = self.cursor_to_dict(txn)

            return self._decode_pushers_rows(rows)

        rows = yield self.runInteraction("get_all_pushers", get_pushers)
        defer.returnValue(rows)

    def get_pushers_stream_token(self):
        return self._pushers_id_gen.get_current_token()

    def get_all_updated_pushers(self, last_id, current_id, limit):
        if last_id == current_id:
            return defer.succeed(([], []))

        def get_all_updated_pushers_txn(txn):
            sql = (
                "SELECT id, user_name, access_token, profile_tag, kind,"
                " app_id, app_display_name, device_display_name, pushkey, ts,"
                " lang, data"
                " FROM pushers"
                " WHERE ? < id AND id <= ?"
                " ORDER BY id ASC LIMIT ?"
            )
            txn.execute(sql, (last_id, current_id, limit))
            updated = txn.fetchall()

            sql = (
                "SELECT stream_id, user_id, app_id, pushkey"
                " FROM deleted_pushers"
                " WHERE ? < stream_id AND stream_id <= ?"
                " ORDER BY stream_id ASC LIMIT ?"
            )
            txn.execute(sql, (last_id, current_id, limit))
            deleted = txn.fetchall()

            return (updated, deleted)
        return self.runInteraction(
            "get_all_updated_pushers", get_all_updated_pushers_txn
        )

    @cachedInlineCallbacks(num_args=1, max_entries=15000)
    def get_if_user_has_pusher(self, user_id):
        # This only exists for the cachedList decorator
        raise NotImplementedError()

    @cachedList(cached_method_name="get_if_user_has_pusher",
                list_name="user_ids", num_args=1, inlineCallbacks=True)
    def get_if_users_have_pushers(self, user_ids):
        rows = yield self._simple_select_many_batch(
            table='pushers',
            column='user_name',
            iterable=user_ids,
            retcols=['user_name'],
            desc='get_if_users_have_pushers'
        )

        result = {user_id: False for user_id in user_ids}
        result.update({r['user_name']: True for r in rows})

        defer.returnValue(result)

    @defer.inlineCallbacks
    def add_pusher(self, user_id, access_token, kind, app_id,
                   app_display_name, device_display_name,
                   pushkey, pushkey_ts, lang, data, last_stream_ordering,
                   profile_tag=""):
        with self._pushers_id_gen.get_next() as stream_id:
            def f(txn):
                newly_inserted = self._simple_upsert_txn(
                    txn,
                    "pushers",
                    {
                        "app_id": app_id,
                        "pushkey": pushkey,
                        "user_name": user_id,
                    },
                    {
                        "access_token": access_token,
                        "kind": kind,
                        "app_display_name": app_display_name,
                        "device_display_name": device_display_name,
                        "ts": pushkey_ts,
                        "lang": lang,
                        "data": encode_canonical_json(data),
                        "last_stream_ordering": last_stream_ordering,
                        "profile_tag": profile_tag,
                        "id": stream_id,
                    },
                )
                if newly_inserted:
                    # get_if_user_has_pusher only cares if the user has
                    # at least *one* pusher.
                    txn.call_after(self.get_if_user_has_pusher.invalidate, (user_id,))

            yield self.runInteraction("add_pusher", f)

    @defer.inlineCallbacks
    def delete_pusher_by_app_id_pushkey_user_id(self, app_id, pushkey, user_id):
        def delete_pusher_txn(txn, stream_id):
            txn.call_after(self.get_if_user_has_pusher.invalidate, (user_id,))

            self._simple_delete_one_txn(
                txn,
                "pushers",
                {"app_id": app_id, "pushkey": pushkey, "user_name": user_id}
            )
            self._simple_upsert_txn(
                txn,
                "deleted_pushers",
                {"app_id": app_id, "pushkey": pushkey, "user_id": user_id},
                {"stream_id": stream_id},
            )

        with self._pushers_id_gen.get_next() as stream_id:
            yield self.runInteraction(
                "delete_pusher", delete_pusher_txn, stream_id
            )

    @defer.inlineCallbacks
    def update_pusher_last_stream_ordering(self, app_id, pushkey, user_id,
                                           last_stream_ordering):
        yield self._simple_update_one(
            "pushers",
            {'app_id': app_id, 'pushkey': pushkey, 'user_name': user_id},
            {'last_stream_ordering': last_stream_ordering},
            desc="update_pusher_last_stream_ordering",
        )

    @defer.inlineCallbacks
    def update_pusher_last_stream_ordering_and_success(self, app_id, pushkey,
                                                       user_id,
                                                       last_stream_ordering,
                                                       last_success):
        yield self._simple_update_one(
            "pushers",
            {'app_id': app_id, 'pushkey': pushkey, 'user_name': user_id},
            {
                'last_stream_ordering': last_stream_ordering,
                'last_success': last_success
            },
            desc="update_pusher_last_stream_ordering_and_success",
        )

    @defer.inlineCallbacks
    def update_pusher_failing_since(self, app_id, pushkey, user_id,
                                    failing_since):
        yield self._simple_update_one(
            "pushers",
            {'app_id': app_id, 'pushkey': pushkey, 'user_name': user_id},
            {'failing_since': failing_since},
            desc="update_pusher_failing_since",
        )

    @defer.inlineCallbacks
    def get_throttle_params_by_room(self, pusher_id):
        res = yield self._simple_select_list(
            "pusher_throttle",
            {"pusher": pusher_id},
            ["room_id", "last_sent_ts", "throttle_ms"],
            desc="get_throttle_params_by_room"
        )

        params_by_room = {}
        for row in res:
            params_by_room[row["room_id"]] = {
                "last_sent_ts": row["last_sent_ts"],
                "throttle_ms": row["throttle_ms"]
            }

        defer.returnValue(params_by_room)

    @defer.inlineCallbacks
    def set_throttle_params(self, pusher_id, room_id, params):
        yield self._simple_upsert(
            "pusher_throttle",
            {"pusher": pusher_id, "room_id": room_id},
            params,
            desc="set_throttle_params"
        )
