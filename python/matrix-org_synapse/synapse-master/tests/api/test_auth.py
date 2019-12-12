# -*- coding: utf-8 -*-
# Copyright 2015 - 2016 OpenMarket Ltd
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

import pymacaroons
from mock import Mock
from twisted.internet import defer

import synapse.handlers.auth
from synapse.api.auth import Auth
from synapse.api.errors import AuthError
from synapse.types import UserID
from tests import unittest
from tests.utils import setup_test_homeserver, mock_getRawHeaders


class TestHandlers(object):
    def __init__(self, hs):
        self.auth_handler = synapse.handlers.auth.AuthHandler(hs)


class AuthTestCase(unittest.TestCase):

    @defer.inlineCallbacks
    def setUp(self):
        self.state_handler = Mock()
        self.store = Mock()

        self.hs = yield setup_test_homeserver(handlers=None)
        self.hs.get_datastore = Mock(return_value=self.store)
        self.hs.handlers = TestHandlers(self.hs)
        self.auth = Auth(self.hs)

        self.test_user = "@foo:bar"
        self.test_token = "_test_token_"

        # this is overridden for the appservice tests
        self.store.get_app_service_by_token = Mock(return_value=None)

    @defer.inlineCallbacks
    def test_get_user_by_req_user_valid_token(self):
        user_info = {
            "name": self.test_user,
            "token_id": "ditto",
            "device_id": "device",
        }
        self.store.get_user_by_access_token = Mock(return_value=user_info)

        request = Mock(args={})
        request.args["access_token"] = [self.test_token]
        request.requestHeaders.getRawHeaders = mock_getRawHeaders()
        requester = yield self.auth.get_user_by_req(request)
        self.assertEquals(requester.user.to_string(), self.test_user)

    def test_get_user_by_req_user_bad_token(self):
        self.store.get_user_by_access_token = Mock(return_value=None)

        request = Mock(args={})
        request.args["access_token"] = [self.test_token]
        request.requestHeaders.getRawHeaders = mock_getRawHeaders()
        d = self.auth.get_user_by_req(request)
        self.failureResultOf(d, AuthError)

    def test_get_user_by_req_user_missing_token(self):
        user_info = {
            "name": self.test_user,
            "token_id": "ditto",
        }
        self.store.get_user_by_access_token = Mock(return_value=user_info)

        request = Mock(args={})
        request.requestHeaders.getRawHeaders = mock_getRawHeaders()
        d = self.auth.get_user_by_req(request)
        self.failureResultOf(d, AuthError)

    @defer.inlineCallbacks
    def test_get_user_by_req_appservice_valid_token(self):
        app_service = Mock(token="foobar", url="a_url", sender=self.test_user)
        self.store.get_app_service_by_token = Mock(return_value=app_service)
        self.store.get_user_by_access_token = Mock(return_value=None)

        request = Mock(args={})
        request.args["access_token"] = [self.test_token]
        request.requestHeaders.getRawHeaders = mock_getRawHeaders()
        requester = yield self.auth.get_user_by_req(request)
        self.assertEquals(requester.user.to_string(), self.test_user)

    def test_get_user_by_req_appservice_bad_token(self):
        self.store.get_app_service_by_token = Mock(return_value=None)
        self.store.get_user_by_access_token = Mock(return_value=None)

        request = Mock(args={})
        request.args["access_token"] = [self.test_token]
        request.requestHeaders.getRawHeaders = mock_getRawHeaders()
        d = self.auth.get_user_by_req(request)
        self.failureResultOf(d, AuthError)

    def test_get_user_by_req_appservice_missing_token(self):
        app_service = Mock(token="foobar", url="a_url", sender=self.test_user)
        self.store.get_app_service_by_token = Mock(return_value=app_service)
        self.store.get_user_by_access_token = Mock(return_value=None)

        request = Mock(args={})
        request.requestHeaders.getRawHeaders = mock_getRawHeaders()
        d = self.auth.get_user_by_req(request)
        self.failureResultOf(d, AuthError)

    @defer.inlineCallbacks
    def test_get_user_by_req_appservice_valid_token_valid_user_id(self):
        masquerading_user_id = "@doppelganger:matrix.org"
        app_service = Mock(token="foobar", url="a_url", sender=self.test_user)
        app_service.is_interested_in_user = Mock(return_value=True)
        self.store.get_app_service_by_token = Mock(return_value=app_service)
        self.store.get_user_by_access_token = Mock(return_value=None)

        request = Mock(args={})
        request.args["access_token"] = [self.test_token]
        request.args["user_id"] = [masquerading_user_id]
        request.requestHeaders.getRawHeaders = mock_getRawHeaders()
        requester = yield self.auth.get_user_by_req(request)
        self.assertEquals(requester.user.to_string(), masquerading_user_id)

    def test_get_user_by_req_appservice_valid_token_bad_user_id(self):
        masquerading_user_id = "@doppelganger:matrix.org"
        app_service = Mock(token="foobar", url="a_url", sender=self.test_user)
        app_service.is_interested_in_user = Mock(return_value=False)
        self.store.get_app_service_by_token = Mock(return_value=app_service)
        self.store.get_user_by_access_token = Mock(return_value=None)

        request = Mock(args={})
        request.args["access_token"] = [self.test_token]
        request.args["user_id"] = [masquerading_user_id]
        request.requestHeaders.getRawHeaders = mock_getRawHeaders()
        d = self.auth.get_user_by_req(request)
        self.failureResultOf(d, AuthError)

    @defer.inlineCallbacks
    def test_get_user_from_macaroon(self):
        # TODO(danielwh): Remove this mock when we remove the
        # get_user_by_access_token fallback.
        self.store.get_user_by_access_token = Mock(
            return_value={
                "name": "@baldrick:matrix.org",
                "device_id": "device",
            }
        )

        user_id = "@baldrick:matrix.org"
        macaroon = pymacaroons.Macaroon(
            location=self.hs.config.server_name,
            identifier="key",
            key=self.hs.config.macaroon_secret_key)
        macaroon.add_first_party_caveat("gen = 1")
        macaroon.add_first_party_caveat("type = access")
        macaroon.add_first_party_caveat("user_id = %s" % (user_id,))
        user_info = yield self.auth.get_user_by_access_token(macaroon.serialize())
        user = user_info["user"]
        self.assertEqual(UserID.from_string(user_id), user)

        # TODO: device_id should come from the macaroon, but currently comes
        # from the db.
        self.assertEqual(user_info["device_id"], "device")

    @defer.inlineCallbacks
    def test_get_guest_user_from_macaroon(self):
        self.store.get_user_by_id = Mock(return_value={
            "is_guest": True,
        })

        user_id = "@baldrick:matrix.org"
        macaroon = pymacaroons.Macaroon(
            location=self.hs.config.server_name,
            identifier="key",
            key=self.hs.config.macaroon_secret_key)
        macaroon.add_first_party_caveat("gen = 1")
        macaroon.add_first_party_caveat("type = access")
        macaroon.add_first_party_caveat("user_id = %s" % (user_id,))
        macaroon.add_first_party_caveat("guest = true")
        serialized = macaroon.serialize()

        user_info = yield self.auth.get_user_by_access_token(serialized)
        user = user_info["user"]
        is_guest = user_info["is_guest"]
        self.assertEqual(UserID.from_string(user_id), user)
        self.assertTrue(is_guest)
        self.store.get_user_by_id.assert_called_with(user_id)

    @defer.inlineCallbacks
    def test_get_user_from_macaroon_user_db_mismatch(self):
        self.store.get_user_by_access_token = Mock(
            return_value={"name": "@percy:matrix.org"}
        )

        user = "@baldrick:matrix.org"
        macaroon = pymacaroons.Macaroon(
            location=self.hs.config.server_name,
            identifier="key",
            key=self.hs.config.macaroon_secret_key)
        macaroon.add_first_party_caveat("gen = 1")
        macaroon.add_first_party_caveat("type = access")
        macaroon.add_first_party_caveat("user_id = %s" % (user,))
        with self.assertRaises(AuthError) as cm:
            yield self.auth.get_user_by_access_token(macaroon.serialize())
        self.assertEqual(401, cm.exception.code)
        self.assertIn("User mismatch", cm.exception.msg)

    @defer.inlineCallbacks
    def test_get_user_from_macaroon_missing_caveat(self):
        # TODO(danielwh): Remove this mock when we remove the
        # get_user_by_access_token fallback.
        self.store.get_user_by_access_token = Mock(
            return_value={"name": "@baldrick:matrix.org"}
        )

        macaroon = pymacaroons.Macaroon(
            location=self.hs.config.server_name,
            identifier="key",
            key=self.hs.config.macaroon_secret_key)
        macaroon.add_first_party_caveat("gen = 1")
        macaroon.add_first_party_caveat("type = access")

        with self.assertRaises(AuthError) as cm:
            yield self.auth.get_user_by_access_token(macaroon.serialize())
        self.assertEqual(401, cm.exception.code)
        self.assertIn("No user caveat", cm.exception.msg)

    @defer.inlineCallbacks
    def test_get_user_from_macaroon_wrong_key(self):
        # TODO(danielwh): Remove this mock when we remove the
        # get_user_by_access_token fallback.
        self.store.get_user_by_access_token = Mock(
            return_value={"name": "@baldrick:matrix.org"}
        )

        user = "@baldrick:matrix.org"
        macaroon = pymacaroons.Macaroon(
            location=self.hs.config.server_name,
            identifier="key",
            key=self.hs.config.macaroon_secret_key + "wrong")
        macaroon.add_first_party_caveat("gen = 1")
        macaroon.add_first_party_caveat("type = access")
        macaroon.add_first_party_caveat("user_id = %s" % (user,))

        with self.assertRaises(AuthError) as cm:
            yield self.auth.get_user_by_access_token(macaroon.serialize())
        self.assertEqual(401, cm.exception.code)
        self.assertIn("Invalid macaroon", cm.exception.msg)

    @defer.inlineCallbacks
    def test_get_user_from_macaroon_unknown_caveat(self):
        # TODO(danielwh): Remove this mock when we remove the
        # get_user_by_access_token fallback.
        self.store.get_user_by_access_token = Mock(
            return_value={"name": "@baldrick:matrix.org"}
        )

        user = "@baldrick:matrix.org"
        macaroon = pymacaroons.Macaroon(
            location=self.hs.config.server_name,
            identifier="key",
            key=self.hs.config.macaroon_secret_key)
        macaroon.add_first_party_caveat("gen = 1")
        macaroon.add_first_party_caveat("type = access")
        macaroon.add_first_party_caveat("user_id = %s" % (user,))
        macaroon.add_first_party_caveat("cunning > fox")

        with self.assertRaises(AuthError) as cm:
            yield self.auth.get_user_by_access_token(macaroon.serialize())
        self.assertEqual(401, cm.exception.code)
        self.assertIn("Invalid macaroon", cm.exception.msg)

    @defer.inlineCallbacks
    def test_get_user_from_macaroon_expired(self):
        # TODO(danielwh): Remove this mock when we remove the
        # get_user_by_access_token fallback.
        self.store.get_user_by_access_token = Mock(
            return_value={"name": "@baldrick:matrix.org"}
        )

        self.store.get_user_by_access_token = Mock(
            return_value={"name": "@baldrick:matrix.org"}
        )

        user = "@baldrick:matrix.org"
        macaroon = pymacaroons.Macaroon(
            location=self.hs.config.server_name,
            identifier="key",
            key=self.hs.config.macaroon_secret_key)
        macaroon.add_first_party_caveat("gen = 1")
        macaroon.add_first_party_caveat("type = access")
        macaroon.add_first_party_caveat("user_id = %s" % (user,))
        macaroon.add_first_party_caveat("time < -2000")  # ms

        self.hs.clock.now = 5000  # seconds
        self.hs.config.expire_access_token = True
        # yield self.auth.get_user_by_access_token(macaroon.serialize())
        # TODO(daniel): Turn on the check that we validate expiration, when we
        # validate expiration (and remove the above line, which will start
        # throwing).
        with self.assertRaises(AuthError) as cm:
            yield self.auth.get_user_by_access_token(macaroon.serialize())
        self.assertEqual(401, cm.exception.code)
        self.assertIn("Invalid macaroon", cm.exception.msg)

    @defer.inlineCallbacks
    def test_get_user_from_macaroon_with_valid_duration(self):
        # TODO(danielwh): Remove this mock when we remove the
        # get_user_by_access_token fallback.
        self.store.get_user_by_access_token = Mock(
            return_value={"name": "@baldrick:matrix.org"}
        )

        self.store.get_user_by_access_token = Mock(
            return_value={"name": "@baldrick:matrix.org"}
        )

        user_id = "@baldrick:matrix.org"
        macaroon = pymacaroons.Macaroon(
            location=self.hs.config.server_name,
            identifier="key",
            key=self.hs.config.macaroon_secret_key)
        macaroon.add_first_party_caveat("gen = 1")
        macaroon.add_first_party_caveat("type = access")
        macaroon.add_first_party_caveat("user_id = %s" % (user_id,))
        macaroon.add_first_party_caveat("time < 900000000")  # ms

        self.hs.clock.now = 5000  # seconds
        self.hs.config.expire_access_token = True

        user_info = yield self.auth.get_user_by_access_token(macaroon.serialize())
        user = user_info["user"]
        self.assertEqual(UserID.from_string(user_id), user)

    @defer.inlineCallbacks
    def test_cannot_use_regular_token_as_guest(self):
        USER_ID = "@percy:matrix.org"
        self.store.add_access_token_to_user = Mock()

        token = yield self.hs.handlers.auth_handler.issue_access_token(
            USER_ID, "DEVICE"
        )
        self.store.add_access_token_to_user.assert_called_with(
            USER_ID, token, "DEVICE"
        )

        def get_user(tok):
            if token != tok:
                return None
            return {
                "name": USER_ID,
                "is_guest": False,
                "token_id": 1234,
                "device_id": "DEVICE",
            }
        self.store.get_user_by_access_token = get_user
        self.store.get_user_by_id = Mock(return_value={
            "is_guest": False,
        })

        # check the token works
        request = Mock(args={})
        request.args["access_token"] = [token]
        request.requestHeaders.getRawHeaders = mock_getRawHeaders()
        requester = yield self.auth.get_user_by_req(request, allow_guest=True)
        self.assertEqual(UserID.from_string(USER_ID), requester.user)
        self.assertFalse(requester.is_guest)

        # add an is_guest caveat
        mac = pymacaroons.Macaroon.deserialize(token)
        mac.add_first_party_caveat("guest = true")
        guest_tok = mac.serialize()

        # the token should *not* work now
        request = Mock(args={})
        request.args["access_token"] = [guest_tok]
        request.requestHeaders.getRawHeaders = mock_getRawHeaders()

        with self.assertRaises(AuthError) as cm:
            yield self.auth.get_user_by_req(request, allow_guest=True)

        self.assertEqual(401, cm.exception.code)
        self.assertEqual("Guest access token used for regular user", cm.exception.msg)

        self.store.get_user_by_id.assert_called_with(USER_ID)
