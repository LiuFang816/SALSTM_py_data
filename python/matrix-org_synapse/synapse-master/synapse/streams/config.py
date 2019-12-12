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

from synapse.api.errors import SynapseError
from synapse.types import StreamToken

import logging


logger = logging.getLogger(__name__)


MAX_LIMIT = 1000


class SourcePaginationConfig(object):

    """A configuration object which stores pagination parameters for a
    specific event source."""

    def __init__(self, from_key=None, to_key=None, direction='f',
                 limit=None):
        self.from_key = from_key
        self.to_key = to_key
        self.direction = 'f' if direction == 'f' else 'b'
        self.limit = min(int(limit), MAX_LIMIT) if limit is not None else None

    def __repr__(self):
        return (
            "StreamConfig(from_key=%r, to_key=%r, direction=%r, limit=%r)"
        ) % (self.from_key, self.to_key, self.direction, self.limit)


class PaginationConfig(object):

    """A configuration object which stores pagination parameters."""

    def __init__(self, from_token=None, to_token=None, direction='f',
                 limit=None):
        self.from_token = from_token
        self.to_token = to_token
        self.direction = 'f' if direction == 'f' else 'b'
        self.limit = min(int(limit), MAX_LIMIT) if limit is not None else None

    @classmethod
    def from_request(cls, request, raise_invalid_params=True,
                     default_limit=None):
        def get_param(name, default=None):
            lst = request.args.get(name, [])
            if len(lst) > 1:
                raise SynapseError(
                    400, "%s must be specified only once" % (name,)
                )
            elif len(lst) == 1:
                return lst[0]
            else:
                return default

        direction = get_param("dir", 'f')
        if direction not in ['f', 'b']:
            raise SynapseError(400, "'dir' parameter is invalid.")

        from_tok = get_param("from")
        to_tok = get_param("to")

        try:
            if from_tok == "END":
                from_tok = None  # For backwards compat.
            elif from_tok:
                from_tok = StreamToken.from_string(from_tok)
        except:
            raise SynapseError(400, "'from' paramater is invalid")

        try:
            if to_tok:
                to_tok = StreamToken.from_string(to_tok)
        except:
            raise SynapseError(400, "'to' paramater is invalid")

        limit = get_param("limit", None)
        if limit is not None and not limit.isdigit():
            raise SynapseError(400, "'limit' parameter must be an integer.")

        if limit is None:
            limit = default_limit

        try:
            return PaginationConfig(from_tok, to_tok, direction, limit)
        except:
            logger.exception("Failed to create pagination config")
            raise SynapseError(400, "Invalid request.")

    def __repr__(self):
        return (
            "PaginationConfig(from_tok=%r, to_tok=%r,"
            " direction=%r, limit=%r)"
        ) % (self.from_token, self.to_token, self.direction, self.limit)

    def get_source_config(self, source_name):
        keyname = "%s_key" % source_name

        return SourcePaginationConfig(
            from_key=getattr(self.from_token, keyname),
            to_key=getattr(self.to_token, keyname) if self.to_token else None,
            direction=self.direction,
            limit=self.limit,
        )
