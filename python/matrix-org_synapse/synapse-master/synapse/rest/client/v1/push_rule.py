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

from synapse.api.errors import (
    SynapseError, UnrecognizedRequestError, NotFoundError, StoreError
)
from .base import ClientV1RestServlet, client_path_patterns
from synapse.storage.push_rule import (
    InconsistentRuleException, RuleNotFoundException
)
from synapse.push.clientformat import format_push_rules_for_user
from synapse.push.baserules import BASE_RULE_IDS
from synapse.push.rulekinds import PRIORITY_CLASS_MAP
from synapse.http.servlet import parse_json_value_from_request


class PushRuleRestServlet(ClientV1RestServlet):
    PATTERNS = client_path_patterns("/pushrules/.*$")
    SLIGHTLY_PEDANTIC_TRAILING_SLASH_ERROR = (
        "Unrecognised request: You probably wanted a trailing slash")

    def __init__(self, hs):
        super(PushRuleRestServlet, self).__init__(hs)
        self.store = hs.get_datastore()
        self.notifier = hs.get_notifier()

    @defer.inlineCallbacks
    def on_PUT(self, request):
        spec = _rule_spec_from_path(request.postpath)
        try:
            priority_class = _priority_class_from_spec(spec)
        except InvalidRuleException as e:
            raise SynapseError(400, e.message)

        requester = yield self.auth.get_user_by_req(request)

        if '/' in spec['rule_id'] or '\\' in spec['rule_id']:
            raise SynapseError(400, "rule_id may not contain slashes")

        content = parse_json_value_from_request(request)

        user_id = requester.user.to_string()

        if 'attr' in spec:
            yield self.set_rule_attr(user_id, spec, content)
            self.notify_user(user_id)
            defer.returnValue((200, {}))

        if spec['rule_id'].startswith('.'):
            # Rule ids starting with '.' are reserved for server default rules.
            raise SynapseError(400, "cannot add new rule_ids that start with '.'")

        try:
            (conditions, actions) = _rule_tuple_from_request_object(
                spec['template'],
                spec['rule_id'],
                content,
            )
        except InvalidRuleException as e:
            raise SynapseError(400, e.message)

        before = request.args.get("before", None)
        if before:
            before = _namespaced_rule_id(spec, before[0])

        after = request.args.get("after", None)
        if after:
            after = _namespaced_rule_id(spec, after[0])

        try:
            yield self.store.add_push_rule(
                user_id=user_id,
                rule_id=_namespaced_rule_id_from_spec(spec),
                priority_class=priority_class,
                conditions=conditions,
                actions=actions,
                before=before,
                after=after
            )
            self.notify_user(user_id)
        except InconsistentRuleException as e:
            raise SynapseError(400, e.message)
        except RuleNotFoundException as e:
            raise SynapseError(400, e.message)

        defer.returnValue((200, {}))

    @defer.inlineCallbacks
    def on_DELETE(self, request):
        spec = _rule_spec_from_path(request.postpath)

        requester = yield self.auth.get_user_by_req(request)
        user_id = requester.user.to_string()

        namespaced_rule_id = _namespaced_rule_id_from_spec(spec)

        try:
            yield self.store.delete_push_rule(
                user_id, namespaced_rule_id
            )
            self.notify_user(user_id)
            defer.returnValue((200, {}))
        except StoreError as e:
            if e.code == 404:
                raise NotFoundError()
            else:
                raise

    @defer.inlineCallbacks
    def on_GET(self, request):
        requester = yield self.auth.get_user_by_req(request)
        user_id = requester.user.to_string()

        # we build up the full structure and then decide which bits of it
        # to send which means doing unnecessary work sometimes but is
        # is probably not going to make a whole lot of difference
        rules = yield self.store.get_push_rules_for_user(user_id)

        rules = format_push_rules_for_user(requester.user, rules)

        path = request.postpath[1:]

        if path == []:
            # we're a reference impl: pedantry is our job.
            raise UnrecognizedRequestError(
                PushRuleRestServlet.SLIGHTLY_PEDANTIC_TRAILING_SLASH_ERROR
            )

        if path[0] == '':
            defer.returnValue((200, rules))
        elif path[0] == 'global':
            path = path[1:]
            result = _filter_ruleset_with_path(rules['global'], path)
            defer.returnValue((200, result))
        else:
            raise UnrecognizedRequestError()

    def on_OPTIONS(self, _):
        return 200, {}

    def notify_user(self, user_id):
        stream_id, _ = self.store.get_push_rules_stream_token()
        self.notifier.on_new_event(
            "push_rules_key", stream_id, users=[user_id]
        )

    def set_rule_attr(self, user_id, spec, val):
        if spec['attr'] == 'enabled':
            if isinstance(val, dict) and "enabled" in val:
                val = val["enabled"]
            if not isinstance(val, bool):
                # Legacy fallback
                # This should *actually* take a dict, but many clients pass
                # bools directly, so let's not break them.
                raise SynapseError(400, "Value for 'enabled' must be boolean")
            namespaced_rule_id = _namespaced_rule_id_from_spec(spec)
            return self.store.set_push_rule_enabled(
                user_id, namespaced_rule_id, val
            )
        elif spec['attr'] == 'actions':
            actions = val.get('actions')
            _check_actions(actions)
            namespaced_rule_id = _namespaced_rule_id_from_spec(spec)
            rule_id = spec['rule_id']
            is_default_rule = rule_id.startswith(".")
            if is_default_rule:
                if namespaced_rule_id not in BASE_RULE_IDS:
                    raise SynapseError(404, "Unknown rule %r" % (namespaced_rule_id,))
            return self.store.set_push_rule_actions(
                user_id, namespaced_rule_id, actions, is_default_rule
            )
        else:
            raise UnrecognizedRequestError()


def _rule_spec_from_path(path):
    if len(path) < 2:
        raise UnrecognizedRequestError()
    if path[0] != 'pushrules':
        raise UnrecognizedRequestError()

    scope = path[1]
    path = path[2:]
    if scope != 'global':
        raise UnrecognizedRequestError()

    if len(path) == 0:
        raise UnrecognizedRequestError()

    template = path[0]
    path = path[1:]

    if len(path) == 0 or len(path[0]) == 0:
        raise UnrecognizedRequestError()

    rule_id = path[0]

    spec = {
        'scope': scope,
        'template': template,
        'rule_id': rule_id
    }

    path = path[1:]

    if len(path) > 0 and len(path[0]) > 0:
        spec['attr'] = path[0]

    return spec


def _rule_tuple_from_request_object(rule_template, rule_id, req_obj):
    if rule_template in ['override', 'underride']:
        if 'conditions' not in req_obj:
            raise InvalidRuleException("Missing 'conditions'")
        conditions = req_obj['conditions']
        for c in conditions:
            if 'kind' not in c:
                raise InvalidRuleException("Condition without 'kind'")
    elif rule_template == 'room':
        conditions = [{
            'kind': 'event_match',
            'key': 'room_id',
            'pattern': rule_id
        }]
    elif rule_template == 'sender':
        conditions = [{
            'kind': 'event_match',
            'key': 'user_id',
            'pattern': rule_id
        }]
    elif rule_template == 'content':
        if 'pattern' not in req_obj:
            raise InvalidRuleException("Content rule missing 'pattern'")
        pat = req_obj['pattern']

        conditions = [{
            'kind': 'event_match',
            'key': 'content.body',
            'pattern': pat
        }]
    else:
        raise InvalidRuleException("Unknown rule template: %s" % (rule_template,))

    if 'actions' not in req_obj:
        raise InvalidRuleException("No actions found")
    actions = req_obj['actions']

    _check_actions(actions)

    return conditions, actions


def _check_actions(actions):
    if not isinstance(actions, list):
        raise InvalidRuleException("No actions found")

    for a in actions:
        if a in ['notify', 'dont_notify', 'coalesce']:
            pass
        elif isinstance(a, dict) and 'set_tweak' in a:
            pass
        else:
            raise InvalidRuleException("Unrecognised action")


def _filter_ruleset_with_path(ruleset, path):
    if path == []:
        raise UnrecognizedRequestError(
            PushRuleRestServlet.SLIGHTLY_PEDANTIC_TRAILING_SLASH_ERROR
        )

    if path[0] == '':
        return ruleset
    template_kind = path[0]
    if template_kind not in ruleset:
        raise UnrecognizedRequestError()
    path = path[1:]
    if path == []:
        raise UnrecognizedRequestError(
            PushRuleRestServlet.SLIGHTLY_PEDANTIC_TRAILING_SLASH_ERROR
        )
    if path[0] == '':
        return ruleset[template_kind]
    rule_id = path[0]

    the_rule = None
    for r in ruleset[template_kind]:
        if r['rule_id'] == rule_id:
            the_rule = r
    if the_rule is None:
        raise NotFoundError

    path = path[1:]
    if len(path) == 0:
        return the_rule

    attr = path[0]
    if attr in the_rule:
        # Make sure we return a JSON object as the attribute may be a
        # JSON value.
        return {attr: the_rule[attr]}
    else:
        raise UnrecognizedRequestError()


def _priority_class_from_spec(spec):
    if spec['template'] not in PRIORITY_CLASS_MAP.keys():
        raise InvalidRuleException("Unknown template: %s" % (spec['template']))
    pc = PRIORITY_CLASS_MAP[spec['template']]

    return pc


def _namespaced_rule_id_from_spec(spec):
    return _namespaced_rule_id(spec, spec['rule_id'])


def _namespaced_rule_id(spec, rule_id):
    return "global/%s/%s" % (spec['template'], rule_id)


class InvalidRuleException(Exception):
    pass


def register_servlets(hs, http_server):
    PushRuleRestServlet(hs).register(http_server)
