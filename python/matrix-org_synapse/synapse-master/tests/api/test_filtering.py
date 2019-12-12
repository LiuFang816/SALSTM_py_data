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

from tests import unittest
from twisted.internet import defer

from mock import Mock
from tests.utils import (
    MockHttpResource, DeferredMockCallable, setup_test_homeserver
)

from synapse.api.filtering import Filter
from synapse.events import FrozenEvent

user_localpart = "test_user"


def MockEvent(**kwargs):
    if "event_id" not in kwargs:
        kwargs["event_id"] = "fake_event_id"
    if "type" not in kwargs:
        kwargs["type"] = "fake_type"
    return FrozenEvent(kwargs)


class FilteringTestCase(unittest.TestCase):

    @defer.inlineCallbacks
    def setUp(self):
        self.mock_federation_resource = MockHttpResource()

        self.mock_http_client = Mock(spec=[])
        self.mock_http_client.put_json = DeferredMockCallable()

        hs = yield setup_test_homeserver(
            handlers=None,
            http_client=self.mock_http_client,
            keyring=Mock(),
        )

        self.filtering = hs.get_filtering()

        self.datastore = hs.get_datastore()

    def test_definition_types_works_with_literals(self):
        definition = {
            "types": ["m.room.message", "org.matrix.foo.bar"]
        }
        event = MockEvent(
            sender="@foo:bar",
            type="m.room.message",
            room_id="!foo:bar"
        )

        self.assertTrue(
            Filter(definition).check(event)
        )

    def test_definition_types_works_with_wildcards(self):
        definition = {
            "types": ["m.*", "org.matrix.foo.bar"]
        }
        event = MockEvent(
            sender="@foo:bar",
            type="m.room.message",
            room_id="!foo:bar"
        )
        self.assertTrue(
            Filter(definition).check(event)
        )

    def test_definition_types_works_with_unknowns(self):
        definition = {
            "types": ["m.room.message", "org.matrix.foo.bar"]
        }
        event = MockEvent(
            sender="@foo:bar",
            type="now.for.something.completely.different",
            room_id="!foo:bar"
        )
        self.assertFalse(
            Filter(definition).check(event)
        )

    def test_definition_not_types_works_with_literals(self):
        definition = {
            "not_types": ["m.room.message", "org.matrix.foo.bar"]
        }
        event = MockEvent(
            sender="@foo:bar",
            type="m.room.message",
            room_id="!foo:bar"
        )
        self.assertFalse(
            Filter(definition).check(event)
        )

    def test_definition_not_types_works_with_wildcards(self):
        definition = {
            "not_types": ["m.room.message", "org.matrix.*"]
        }
        event = MockEvent(
            sender="@foo:bar",
            type="org.matrix.custom.event",
            room_id="!foo:bar"
        )
        self.assertFalse(
            Filter(definition).check(event)
        )

    def test_definition_not_types_works_with_unknowns(self):
        definition = {
            "not_types": ["m.*", "org.*"]
        }
        event = MockEvent(
            sender="@foo:bar",
            type="com.nom.nom.nom",
            room_id="!foo:bar"
        )
        self.assertTrue(
            Filter(definition).check(event)
        )

    def test_definition_not_types_takes_priority_over_types(self):
        definition = {
            "not_types": ["m.*", "org.*"],
            "types": ["m.room.message", "m.room.topic"]
        }
        event = MockEvent(
            sender="@foo:bar",
            type="m.room.topic",
            room_id="!foo:bar"
        )
        self.assertFalse(
            Filter(definition).check(event)
        )

    def test_definition_senders_works_with_literals(self):
        definition = {
            "senders": ["@flibble:wibble"]
        }
        event = MockEvent(
            sender="@flibble:wibble",
            type="com.nom.nom.nom",
            room_id="!foo:bar"
        )
        self.assertTrue(
            Filter(definition).check(event)
        )

    def test_definition_senders_works_with_unknowns(self):
        definition = {
            "senders": ["@flibble:wibble"]
        }
        event = MockEvent(
            sender="@challenger:appears",
            type="com.nom.nom.nom",
            room_id="!foo:bar"
        )
        self.assertFalse(
            Filter(definition).check(event)
        )

    def test_definition_not_senders_works_with_literals(self):
        definition = {
            "not_senders": ["@flibble:wibble"]
        }
        event = MockEvent(
            sender="@flibble:wibble",
            type="com.nom.nom.nom",
            room_id="!foo:bar"
        )
        self.assertFalse(
            Filter(definition).check(event)
        )

    def test_definition_not_senders_works_with_unknowns(self):
        definition = {
            "not_senders": ["@flibble:wibble"]
        }
        event = MockEvent(
            sender="@challenger:appears",
            type="com.nom.nom.nom",
            room_id="!foo:bar"
        )
        self.assertTrue(
            Filter(definition).check(event)
        )

    def test_definition_not_senders_takes_priority_over_senders(self):
        definition = {
            "not_senders": ["@misspiggy:muppets"],
            "senders": ["@kermit:muppets", "@misspiggy:muppets"]
        }
        event = MockEvent(
            sender="@misspiggy:muppets",
            type="m.room.topic",
            room_id="!foo:bar"
        )
        self.assertFalse(
            Filter(definition).check(event)
        )

    def test_definition_rooms_works_with_literals(self):
        definition = {
            "rooms": ["!secretbase:unknown"]
        }
        event = MockEvent(
            sender="@foo:bar",
            type="m.room.message",
            room_id="!secretbase:unknown"
        )
        self.assertTrue(
            Filter(definition).check(event)
        )

    def test_definition_rooms_works_with_unknowns(self):
        definition = {
            "rooms": ["!secretbase:unknown"]
        }
        event = MockEvent(
            sender="@foo:bar",
            type="m.room.message",
            room_id="!anothersecretbase:unknown"
        )
        self.assertFalse(
            Filter(definition).check(event)
        )

    def test_definition_not_rooms_works_with_literals(self):
        definition = {
            "not_rooms": ["!anothersecretbase:unknown"]
        }
        event = MockEvent(
            sender="@foo:bar",
            type="m.room.message",
            room_id="!anothersecretbase:unknown"
        )
        self.assertFalse(
            Filter(definition).check(event)
        )

    def test_definition_not_rooms_works_with_unknowns(self):
        definition = {
            "not_rooms": ["!secretbase:unknown"]
        }
        event = MockEvent(
            sender="@foo:bar",
            type="m.room.message",
            room_id="!anothersecretbase:unknown"
        )
        self.assertTrue(
            Filter(definition).check(event)
        )

    def test_definition_not_rooms_takes_priority_over_rooms(self):
        definition = {
            "not_rooms": ["!secretbase:unknown"],
            "rooms": ["!secretbase:unknown"]
        }
        event = MockEvent(
            sender="@foo:bar",
            type="m.room.message",
            room_id="!secretbase:unknown"
        )
        self.assertFalse(
            Filter(definition).check(event)
        )

    def test_definition_combined_event(self):
        definition = {
            "not_senders": ["@misspiggy:muppets"],
            "senders": ["@kermit:muppets"],
            "rooms": ["!stage:unknown"],
            "not_rooms": ["!piggyshouse:muppets"],
            "types": ["m.room.message", "muppets.kermit.*"],
            "not_types": ["muppets.misspiggy.*"]
        }
        event = MockEvent(
            sender="@kermit:muppets",  # yup
            type="m.room.message",  # yup
            room_id="!stage:unknown"  # yup
        )
        self.assertTrue(
            Filter(definition).check(event)
        )

    def test_definition_combined_event_bad_sender(self):
        definition = {
            "not_senders": ["@misspiggy:muppets"],
            "senders": ["@kermit:muppets"],
            "rooms": ["!stage:unknown"],
            "not_rooms": ["!piggyshouse:muppets"],
            "types": ["m.room.message", "muppets.kermit.*"],
            "not_types": ["muppets.misspiggy.*"]
        }
        event = MockEvent(
            sender="@misspiggy:muppets",  # nope
            type="m.room.message",  # yup
            room_id="!stage:unknown"  # yup
        )
        self.assertFalse(
            Filter(definition).check(event)
        )

    def test_definition_combined_event_bad_room(self):
        definition = {
            "not_senders": ["@misspiggy:muppets"],
            "senders": ["@kermit:muppets"],
            "rooms": ["!stage:unknown"],
            "not_rooms": ["!piggyshouse:muppets"],
            "types": ["m.room.message", "muppets.kermit.*"],
            "not_types": ["muppets.misspiggy.*"]
        }
        event = MockEvent(
            sender="@kermit:muppets",  # yup
            type="m.room.message",  # yup
            room_id="!piggyshouse:muppets"  # nope
        )
        self.assertFalse(
            Filter(definition).check(event)
        )

    def test_definition_combined_event_bad_type(self):
        definition = {
            "not_senders": ["@misspiggy:muppets"],
            "senders": ["@kermit:muppets"],
            "rooms": ["!stage:unknown"],
            "not_rooms": ["!piggyshouse:muppets"],
            "types": ["m.room.message", "muppets.kermit.*"],
            "not_types": ["muppets.misspiggy.*"]
        }
        event = MockEvent(
            sender="@kermit:muppets",  # yup
            type="muppets.misspiggy.kisses",  # nope
            room_id="!stage:unknown"  # yup
        )
        self.assertFalse(
            Filter(definition).check(event)
        )

    @defer.inlineCallbacks
    def test_filter_presence_match(self):
        user_filter_json = {
            "presence": {
                "types": ["m.*"]
            }
        }
        filter_id = yield self.datastore.add_user_filter(
            user_localpart=user_localpart,
            user_filter=user_filter_json,
        )
        event = MockEvent(
            sender="@foo:bar",
            type="m.profile",
        )
        events = [event]

        user_filter = yield self.filtering.get_user_filter(
            user_localpart=user_localpart,
            filter_id=filter_id,
        )

        results = user_filter.filter_presence(events=events)
        self.assertEquals(events, results)

    @defer.inlineCallbacks
    def test_filter_presence_no_match(self):
        user_filter_json = {
            "presence": {
                "types": ["m.*"]
            }
        }

        filter_id = yield self.datastore.add_user_filter(
            user_localpart=user_localpart + "2",
            user_filter=user_filter_json,
        )
        event = MockEvent(
            event_id="$asdasd:localhost",
            sender="@foo:bar",
            type="custom.avatar.3d.crazy",
        )
        events = [event]

        user_filter = yield self.filtering.get_user_filter(
            user_localpart=user_localpart + "2",
            filter_id=filter_id,
        )

        results = user_filter.filter_presence(events=events)
        self.assertEquals([], results)

    @defer.inlineCallbacks
    def test_filter_room_state_match(self):
        user_filter_json = {
            "room": {
                "state": {
                    "types": ["m.*"]
                }
            }
        }
        filter_id = yield self.datastore.add_user_filter(
            user_localpart=user_localpart,
            user_filter=user_filter_json,
        )
        event = MockEvent(
            sender="@foo:bar",
            type="m.room.topic",
            room_id="!foo:bar"
        )
        events = [event]

        user_filter = yield self.filtering.get_user_filter(
            user_localpart=user_localpart,
            filter_id=filter_id,
        )

        results = user_filter.filter_room_state(events=events)
        self.assertEquals(events, results)

    @defer.inlineCallbacks
    def test_filter_room_state_no_match(self):
        user_filter_json = {
            "room": {
                "state": {
                    "types": ["m.*"]
                }
            }
        }
        filter_id = yield self.datastore.add_user_filter(
            user_localpart=user_localpart,
            user_filter=user_filter_json,
        )
        event = MockEvent(
            sender="@foo:bar",
            type="org.matrix.custom.event",
            room_id="!foo:bar"
        )
        events = [event]

        user_filter = yield self.filtering.get_user_filter(
            user_localpart=user_localpart,
            filter_id=filter_id,
        )

        results = user_filter.filter_room_state(events)
        self.assertEquals([], results)

    def test_filter_rooms(self):
        definition = {
            "rooms": ["!allowed:example.com", "!excluded:example.com"],
            "not_rooms": ["!excluded:example.com"],
        }

        room_ids = [
            "!allowed:example.com",  # Allowed because in rooms and not in not_rooms.
            "!excluded:example.com",  # Disallowed because in not_rooms.
            "!not_included:example.com",  # Disallowed because not in rooms.
        ]

        filtered_room_ids = list(Filter(definition).filter_rooms(room_ids))

        self.assertEquals(filtered_room_ids, ["!allowed:example.com"])

    @defer.inlineCallbacks
    def test_add_filter(self):
        user_filter_json = {
            "room": {
                "state": {
                    "types": ["m.*"]
                }
            }
        }

        filter_id = yield self.filtering.add_user_filter(
            user_localpart=user_localpart,
            user_filter=user_filter_json,
        )

        self.assertEquals(filter_id, 0)
        self.assertEquals(user_filter_json, (
            yield self.datastore.get_user_filter(
                user_localpart=user_localpart,
                filter_id=0,
            )
        ))

    @defer.inlineCallbacks
    def test_get_filter(self):
        user_filter_json = {
            "room": {
                "state": {
                    "types": ["m.*"]
                }
            }
        }

        filter_id = yield self.datastore.add_user_filter(
            user_localpart=user_localpart,
            user_filter=user_filter_json,
        )

        filter = yield self.filtering.get_user_filter(
            user_localpart=user_localpart,
            filter_id=filter_id,
        )

        self.assertEquals(filter.get_filter_json(), user_filter_json)

        self.assertRegexpMatches(repr(filter), r"<FilterCollection \{.*\}>")
