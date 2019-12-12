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

from synapse.api.errors import SynapseError, AuthError
from synapse.util.logcontext import preserve_fn
from synapse.util.metrics import Measure
from synapse.util.wheel_timer import WheelTimer
from synapse.types import UserID, get_domain_from_id

import logging

from collections import namedtuple
import ujson as json

logger = logging.getLogger(__name__)


# A tiny object useful for storing a user's membership in a room, as a mapping
# key
RoomMember = namedtuple("RoomMember", ("room_id", "user_id"))


# How often we expect remote servers to resend us presence.
FEDERATION_TIMEOUT = 60 * 1000

# How often to resend typing across federation.
FEDERATION_PING_INTERVAL = 40 * 1000


class TypingHandler(object):
    def __init__(self, hs):
        self.store = hs.get_datastore()
        self.server_name = hs.config.server_name
        self.auth = hs.get_auth()
        self.is_mine_id = hs.is_mine_id
        self.notifier = hs.get_notifier()
        self.state = hs.get_state_handler()

        self.hs = hs

        self.clock = hs.get_clock()
        self.wheel_timer = WheelTimer(bucket_size=5000)

        self.federation = hs.get_federation_sender()

        hs.get_replication_layer().register_edu_handler("m.typing", self._recv_edu)

        hs.get_distributor().observe("user_left_room", self.user_left_room)

        self._member_typing_until = {}  # clock time we expect to stop
        self._member_last_federation_poke = {}

        # map room IDs to serial numbers
        self._room_serials = {}
        self._latest_room_serial = 0
        # map room IDs to sets of users currently typing
        self._room_typing = {}

        self.clock.looping_call(
            self._handle_timeouts,
            5000,
        )

    def _handle_timeouts(self):
        logger.info("Checking for typing timeouts")

        now = self.clock.time_msec()

        members = set(self.wheel_timer.fetch(now))

        for member in members:
            if not self.is_typing(member):
                # Nothing to do if they're no longer typing
                continue

            until = self._member_typing_until.get(member, None)
            if not until or until <= now:
                logger.info("Timing out typing for: %s", member.user_id)
                preserve_fn(self._stopped_typing)(member)
                continue

            # Check if we need to resend a keep alive over federation for this
            # user.
            if self.hs.is_mine_id(member.user_id):
                last_fed_poke = self._member_last_federation_poke.get(member, None)
                if not last_fed_poke or last_fed_poke + FEDERATION_PING_INTERVAL <= now:
                    preserve_fn(self._push_remote)(
                        member=member,
                        typing=True
                    )

            # Add a paranoia timer to ensure that we always have a timer for
            # each person typing.
            self.wheel_timer.insert(
                now=now,
                obj=member,
                then=now + 60 * 1000,
            )

    def is_typing(self, member):
        return member.user_id in self._room_typing.get(member.room_id, [])

    @defer.inlineCallbacks
    def started_typing(self, target_user, auth_user, room_id, timeout):
        target_user_id = target_user.to_string()
        auth_user_id = auth_user.to_string()

        if not self.is_mine_id(target_user_id):
            raise SynapseError(400, "User is not hosted on this Home Server")

        if target_user_id != auth_user_id:
            raise AuthError(400, "Cannot set another user's typing state")

        yield self.auth.check_joined_room(room_id, target_user_id)

        logger.debug(
            "%s has started typing in %s", target_user_id, room_id
        )

        member = RoomMember(room_id=room_id, user_id=target_user_id)

        was_present = member.user_id in self._room_typing.get(room_id, set())

        now = self.clock.time_msec()
        self._member_typing_until[member] = now + timeout

        self.wheel_timer.insert(
            now=now,
            obj=member,
            then=now + timeout,
        )

        if was_present:
            # No point sending another notification
            defer.returnValue(None)

        yield self._push_update(
            member=member,
            typing=True,
        )

    @defer.inlineCallbacks
    def stopped_typing(self, target_user, auth_user, room_id):
        target_user_id = target_user.to_string()
        auth_user_id = auth_user.to_string()

        if not self.is_mine_id(target_user_id):
            raise SynapseError(400, "User is not hosted on this Home Server")

        if target_user_id != auth_user_id:
            raise AuthError(400, "Cannot set another user's typing state")

        yield self.auth.check_joined_room(room_id, target_user_id)

        logger.debug(
            "%s has stopped typing in %s", target_user_id, room_id
        )

        member = RoomMember(room_id=room_id, user_id=target_user_id)

        yield self._stopped_typing(member)

    @defer.inlineCallbacks
    def user_left_room(self, user, room_id):
        user_id = user.to_string()
        if self.is_mine_id(user_id):
            member = RoomMember(room_id=room_id, user_id=user_id)
            yield self._stopped_typing(member)

    @defer.inlineCallbacks
    def _stopped_typing(self, member):
        if member.user_id not in self._room_typing.get(member.room_id, set()):
            # No point
            defer.returnValue(None)

        self._member_typing_until.pop(member, None)
        self._member_last_federation_poke.pop(member, None)

        yield self._push_update(
            member=member,
            typing=False,
        )

    @defer.inlineCallbacks
    def _push_update(self, member, typing):
        if self.hs.is_mine_id(member.user_id):
            # Only send updates for changes to our own users.
            yield self._push_remote(member, typing)

        self._push_update_local(
            member=member,
            typing=typing
        )

    @defer.inlineCallbacks
    def _push_remote(self, member, typing):
        users = yield self.state.get_current_user_in_room(member.room_id)
        self._member_last_federation_poke[member] = self.clock.time_msec()

        now = self.clock.time_msec()
        self.wheel_timer.insert(
            now=now,
            obj=member,
            then=now + FEDERATION_PING_INTERVAL,
        )

        for domain in set(get_domain_from_id(u) for u in users):
            if domain != self.server_name:
                self.federation.send_edu(
                    destination=domain,
                    edu_type="m.typing",
                    content={
                        "room_id": member.room_id,
                        "user_id": member.user_id,
                        "typing": typing,
                    },
                    key=member,
                )

    @defer.inlineCallbacks
    def _recv_edu(self, origin, content):
        room_id = content["room_id"]
        user_id = content["user_id"]

        member = RoomMember(user_id=user_id, room_id=room_id)

        # Check that the string is a valid user id
        user = UserID.from_string(user_id)

        if user.domain != origin:
            logger.info(
                "Got typing update from %r with bad 'user_id': %r",
                origin, user_id,
            )
            return

        users = yield self.state.get_current_user_in_room(room_id)
        domains = set(get_domain_from_id(u) for u in users)

        if self.server_name in domains:
            logger.info("Got typing update from %s: %r", user_id, content)
            now = self.clock.time_msec()
            self._member_typing_until[member] = now + FEDERATION_TIMEOUT
            self.wheel_timer.insert(
                now=now,
                obj=member,
                then=now + FEDERATION_TIMEOUT,
            )
            self._push_update_local(
                member=member,
                typing=content["typing"]
            )

    def _push_update_local(self, member, typing):
        room_set = self._room_typing.setdefault(member.room_id, set())
        if typing:
            room_set.add(member.user_id)
        else:
            room_set.discard(member.user_id)

        self._latest_room_serial += 1
        self._room_serials[member.room_id] = self._latest_room_serial

        self.notifier.on_new_event(
            "typing_key", self._latest_room_serial, rooms=[member.room_id]
        )

    def get_all_typing_updates(self, last_id, current_id):
        # TODO: Work out a way to do this without scanning the entire state.
        if last_id == current_id:
            return []

        rows = []
        for room_id, serial in self._room_serials.items():
            if last_id < serial and serial <= current_id:
                typing = self._room_typing[room_id]
                typing_bytes = json.dumps(list(typing), ensure_ascii=False)
                rows.append((serial, room_id, typing_bytes))
        rows.sort()
        return rows


class TypingNotificationEventSource(object):
    def __init__(self, hs):
        self.hs = hs
        self.clock = hs.get_clock()
        # We can't call get_typing_handler here because there's a cycle:
        #
        #   Typing -> Notifier -> TypingNotificationEventSource -> Typing
        #
        self.get_typing_handler = hs.get_typing_handler

    def _make_event_for(self, room_id):
        typing = self.get_typing_handler()._room_typing[room_id]
        return {
            "type": "m.typing",
            "room_id": room_id,
            "content": {
                "user_ids": list(typing),
            },
        }

    def get_new_events(self, from_key, room_ids, **kwargs):
        with Measure(self.clock, "typing.get_new_events"):
            from_key = int(from_key)
            handler = self.get_typing_handler()

            events = []
            for room_id in room_ids:
                if room_id not in handler._room_serials:
                    continue
                if handler._room_serials[room_id] <= from_key:
                    continue

                events.append(self._make_event_for(room_id))

            return events, handler._latest_room_serial

    def get_current_key(self):
        return self.get_typing_handler()._latest_room_serial

    def get_pagination_rows(self, user, pagination_config, key):
        return ([], pagination_config.from_key)
