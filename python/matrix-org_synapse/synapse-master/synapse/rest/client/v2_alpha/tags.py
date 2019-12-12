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

from ._base import client_v2_patterns

from synapse.http.servlet import RestServlet, parse_json_object_from_request
from synapse.api.errors import AuthError

from twisted.internet import defer

import logging

logger = logging.getLogger(__name__)


class TagListServlet(RestServlet):
    """
    GET /user/{user_id}/rooms/{room_id}/tags HTTP/1.1
    """
    PATTERNS = client_v2_patterns(
        "/user/(?P<user_id>[^/]*)/rooms/(?P<room_id>[^/]*)/tags"
    )

    def __init__(self, hs):
        super(TagListServlet, self).__init__()
        self.auth = hs.get_auth()
        self.store = hs.get_datastore()

    @defer.inlineCallbacks
    def on_GET(self, request, user_id, room_id):
        requester = yield self.auth.get_user_by_req(request)
        if user_id != requester.user.to_string():
            raise AuthError(403, "Cannot get tags for other users.")

        tags = yield self.store.get_tags_for_room(user_id, room_id)

        defer.returnValue((200, {"tags": tags}))


class TagServlet(RestServlet):
    """
    PUT /user/{user_id}/rooms/{room_id}/tags/{tag} HTTP/1.1
    DELETE /user/{user_id}/rooms/{room_id}/tags/{tag} HTTP/1.1
    """
    PATTERNS = client_v2_patterns(
        "/user/(?P<user_id>[^/]*)/rooms/(?P<room_id>[^/]*)/tags/(?P<tag>[^/]*)"
    )

    def __init__(self, hs):
        super(TagServlet, self).__init__()
        self.auth = hs.get_auth()
        self.store = hs.get_datastore()
        self.notifier = hs.get_notifier()

    @defer.inlineCallbacks
    def on_PUT(self, request, user_id, room_id, tag):
        requester = yield self.auth.get_user_by_req(request)
        if user_id != requester.user.to_string():
            raise AuthError(403, "Cannot add tags for other users.")

        body = parse_json_object_from_request(request)

        max_id = yield self.store.add_tag_to_room(user_id, room_id, tag, body)

        self.notifier.on_new_event(
            "account_data_key", max_id, users=[user_id]
        )

        defer.returnValue((200, {}))

    @defer.inlineCallbacks
    def on_DELETE(self, request, user_id, room_id, tag):
        requester = yield self.auth.get_user_by_req(request)
        if user_id != requester.user.to_string():
            raise AuthError(403, "Cannot add tags for other users.")

        max_id = yield self.store.remove_tag_from_room(user_id, room_id, tag)

        self.notifier.on_new_event(
            "account_data_key", max_id, users=[user_id]
        )

        defer.returnValue((200, {}))


def register_servlets(hs, http_server):
    TagListServlet(hs).register(http_server)
    TagServlet(hs).register(http_server)
