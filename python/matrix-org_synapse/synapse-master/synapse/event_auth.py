# -*- coding: utf-8 -*-
# Copyright 2014 - 2016 OpenMarket Ltd
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

import logging

from canonicaljson import encode_canonical_json
from signedjson.key import decode_verify_key_bytes
from signedjson.sign import verify_signed_json, SignatureVerifyException
from unpaddedbase64 import decode_base64

from synapse.api.constants import EventTypes, Membership, JoinRules
from synapse.api.errors import AuthError, SynapseError, EventSizeError
from synapse.types import UserID, get_domain_from_id

logger = logging.getLogger(__name__)


def check(event, auth_events, do_sig_check=True, do_size_check=True):
    """ Checks if this event is correctly authed.

    Args:
        event: the event being checked.
        auth_events (dict: event-key -> event): the existing room state.


    Returns:
        True if the auth checks pass.
    """
    if do_size_check:
        _check_size_limits(event)

    if not hasattr(event, "room_id"):
        raise AuthError(500, "Event has no room_id: %s" % event)

    if do_sig_check:
        sender_domain = get_domain_from_id(event.sender)
        event_id_domain = get_domain_from_id(event.event_id)

        is_invite_via_3pid = (
            event.type == EventTypes.Member
            and event.membership == Membership.INVITE
            and "third_party_invite" in event.content
        )

        # Check the sender's domain has signed the event
        if not event.signatures.get(sender_domain):
            # We allow invites via 3pid to have a sender from a different
            # HS, as the sender must match the sender of the original
            # 3pid invite. This is checked further down with the
            # other dedicated membership checks.
            if not is_invite_via_3pid:
                raise AuthError(403, "Event not signed by sender's server")

        # Check the event_id's domain has signed the event
        if not event.signatures.get(event_id_domain):
            raise AuthError(403, "Event not signed by sending server")

    if auth_events is None:
        # Oh, we don't know what the state of the room was, so we
        # are trusting that this is allowed (at least for now)
        logger.warn("Trusting event: %s", event.event_id)
        return True

    if event.type == EventTypes.Create:
        room_id_domain = get_domain_from_id(event.room_id)
        if room_id_domain != sender_domain:
            raise AuthError(
                403,
                "Creation event's room_id domain does not match sender's"
            )
        # FIXME
        return True

    creation_event = auth_events.get((EventTypes.Create, ""), None)

    if not creation_event:
        raise SynapseError(
            403,
            "Room %r does not exist" % (event.room_id,)
        )

    creating_domain = get_domain_from_id(event.room_id)
    originating_domain = get_domain_from_id(event.sender)
    if creating_domain != originating_domain:
        if not _can_federate(event, auth_events):
            raise AuthError(
                403,
                "This room has been marked as unfederatable."
            )

    # FIXME: Temp hack
    if event.type == EventTypes.Aliases:
        if not event.is_state():
            raise AuthError(
                403,
                "Alias event must be a state event",
            )
        if not event.state_key:
            raise AuthError(
                403,
                "Alias event must have non-empty state_key"
            )
        sender_domain = get_domain_from_id(event.sender)
        if event.state_key != sender_domain:
            raise AuthError(
                403,
                "Alias event's state_key does not match sender's domain"
            )
        return True

    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(
            "Auth events: %s",
            [a.event_id for a in auth_events.values()]
        )

    if event.type == EventTypes.Member:
        allowed = _is_membership_change_allowed(
            event, auth_events
        )
        if allowed:
            logger.debug("Allowing! %s", event)
        else:
            logger.debug("Denying! %s", event)
        return allowed

    _check_event_sender_in_room(event, auth_events)

    # Special case to allow m.room.third_party_invite events wherever
    # a user is allowed to issue invites.  Fixes
    # https://github.com/vector-im/vector-web/issues/1208 hopefully
    if event.type == EventTypes.ThirdPartyInvite:
        user_level = get_user_power_level(event.user_id, auth_events)
        invite_level = _get_named_level(auth_events, "invite", 0)

        if user_level < invite_level:
            raise AuthError(
                403, (
                    "You cannot issue a third party invite for %s." %
                    (event.content.display_name,)
                )
            )
        else:
            return True

    _can_send_event(event, auth_events)

    if event.type == EventTypes.PowerLevels:
        _check_power_levels(event, auth_events)

    if event.type == EventTypes.Redaction:
        check_redaction(event, auth_events)

    logger.debug("Allowing! %s", event)


def _check_size_limits(event):
    def too_big(field):
        raise EventSizeError("%s too large" % (field,))

    if len(event.user_id) > 255:
        too_big("user_id")
    if len(event.room_id) > 255:
        too_big("room_id")
    if event.is_state() and len(event.state_key) > 255:
        too_big("state_key")
    if len(event.type) > 255:
        too_big("type")
    if len(event.event_id) > 255:
        too_big("event_id")
    if len(encode_canonical_json(event.get_pdu_json())) > 65536:
        too_big("event")


def _can_federate(event, auth_events):
    creation_event = auth_events.get((EventTypes.Create, ""))

    return creation_event.content.get("m.federate", True) is True


def _is_membership_change_allowed(event, auth_events):
    membership = event.content["membership"]

    # Check if this is the room creator joining:
    if len(event.prev_events) == 1 and Membership.JOIN == membership:
        # Get room creation event:
        key = (EventTypes.Create, "", )
        create = auth_events.get(key)
        if create and event.prev_events[0][0] == create.event_id:
            if create.content["creator"] == event.state_key:
                return True

    target_user_id = event.state_key

    creating_domain = get_domain_from_id(event.room_id)
    target_domain = get_domain_from_id(target_user_id)
    if creating_domain != target_domain:
        if not _can_federate(event, auth_events):
            raise AuthError(
                403,
                "This room has been marked as unfederatable."
            )

    # get info about the caller
    key = (EventTypes.Member, event.user_id, )
    caller = auth_events.get(key)

    caller_in_room = caller and caller.membership == Membership.JOIN
    caller_invited = caller and caller.membership == Membership.INVITE

    # get info about the target
    key = (EventTypes.Member, target_user_id, )
    target = auth_events.get(key)

    target_in_room = target and target.membership == Membership.JOIN
    target_banned = target and target.membership == Membership.BAN

    key = (EventTypes.JoinRules, "", )
    join_rule_event = auth_events.get(key)
    if join_rule_event:
        join_rule = join_rule_event.content.get(
            "join_rule", JoinRules.INVITE
        )
    else:
        join_rule = JoinRules.INVITE

    user_level = get_user_power_level(event.user_id, auth_events)
    target_level = get_user_power_level(
        target_user_id, auth_events
    )

    # FIXME (erikj): What should we do here as the default?
    ban_level = _get_named_level(auth_events, "ban", 50)

    logger.debug(
        "_is_membership_change_allowed: %s",
        {
            "caller_in_room": caller_in_room,
            "caller_invited": caller_invited,
            "target_banned": target_banned,
            "target_in_room": target_in_room,
            "membership": membership,
            "join_rule": join_rule,
            "target_user_id": target_user_id,
            "event.user_id": event.user_id,
        }
    )

    if Membership.INVITE == membership and "third_party_invite" in event.content:
        if not _verify_third_party_invite(event, auth_events):
            raise AuthError(403, "You are not invited to this room.")
        if target_banned:
            raise AuthError(
                403, "%s is banned from the room" % (target_user_id,)
            )
        return True

    if Membership.JOIN != membership:
        if (caller_invited
                and Membership.LEAVE == membership
                and target_user_id == event.user_id):
            return True

        if not caller_in_room:  # caller isn't joined
            raise AuthError(
                403,
                "%s not in room %s." % (event.user_id, event.room_id,)
            )

    if Membership.INVITE == membership:
        # TODO (erikj): We should probably handle this more intelligently
        # PRIVATE join rules.

        # Invites are valid iff caller is in the room and target isn't.
        if target_banned:
            raise AuthError(
                403, "%s is banned from the room" % (target_user_id,)
            )
        elif target_in_room:  # the target is already in the room.
            raise AuthError(403, "%s is already in the room." %
                                 target_user_id)
        else:
            invite_level = _get_named_level(auth_events, "invite", 0)

            if user_level < invite_level:
                raise AuthError(
                    403, "You cannot invite user %s." % target_user_id
                )
    elif Membership.JOIN == membership:
        # Joins are valid iff caller == target and they were:
        # invited: They are accepting the invitation
        # joined: It's a NOOP
        if event.user_id != target_user_id:
            raise AuthError(403, "Cannot force another user to join.")
        elif target_banned:
            raise AuthError(403, "You are banned from this room")
        elif join_rule == JoinRules.PUBLIC:
            pass
        elif join_rule == JoinRules.INVITE:
            if not caller_in_room and not caller_invited:
                raise AuthError(403, "You are not invited to this room.")
        else:
            # TODO (erikj): may_join list
            # TODO (erikj): private rooms
            raise AuthError(403, "You are not allowed to join this room")
    elif Membership.LEAVE == membership:
        # TODO (erikj): Implement kicks.
        if target_banned and user_level < ban_level:
            raise AuthError(
                403, "You cannot unban user &s." % (target_user_id,)
            )
        elif target_user_id != event.user_id:
            kick_level = _get_named_level(auth_events, "kick", 50)

            if user_level < kick_level or user_level <= target_level:
                raise AuthError(
                    403, "You cannot kick user %s." % target_user_id
                )
    elif Membership.BAN == membership:
        if user_level < ban_level or user_level <= target_level:
            raise AuthError(403, "You don't have permission to ban")
    else:
        raise AuthError(500, "Unknown membership %s" % membership)

    return True


def _check_event_sender_in_room(event, auth_events):
    key = (EventTypes.Member, event.user_id, )
    member_event = auth_events.get(key)

    return _check_joined_room(
        member_event,
        event.user_id,
        event.room_id
    )


def _check_joined_room(member, user_id, room_id):
    if not member or member.membership != Membership.JOIN:
        raise AuthError(403, "User %s not in room %s (%s)" % (
            user_id, room_id, repr(member)
        ))


def get_send_level(etype, state_key, auth_events):
    key = (EventTypes.PowerLevels, "", )
    send_level_event = auth_events.get(key)
    send_level = None
    if send_level_event:
        send_level = send_level_event.content.get("events", {}).get(
            etype
        )
        if send_level is None:
            if state_key is not None:
                send_level = send_level_event.content.get(
                    "state_default", 50
                )
            else:
                send_level = send_level_event.content.get(
                    "events_default", 0
                )

    if send_level:
        send_level = int(send_level)
    else:
        send_level = 0

    return send_level


def _can_send_event(event, auth_events):
    send_level = get_send_level(
        event.type, event.get("state_key", None), auth_events
    )
    user_level = get_user_power_level(event.user_id, auth_events)

    if user_level < send_level:
        raise AuthError(
            403,
            "You don't have permission to post that to the room. " +
            "user_level (%d) < send_level (%d)" % (user_level, send_level)
        )

    # Check state_key
    if hasattr(event, "state_key"):
        if event.state_key.startswith("@"):
            if event.state_key != event.user_id:
                raise AuthError(
                    403,
                    "You are not allowed to set others state"
                )

    return True


def check_redaction(event, auth_events):
    """Check whether the event sender is allowed to redact the target event.

    Returns:
        True if the the sender is allowed to redact the target event if the
        target event was created by them.
        False if the sender is allowed to redact the target event with no
        further checks.

    Raises:
        AuthError if the event sender is definitely not allowed to redact
        the target event.
    """
    user_level = get_user_power_level(event.user_id, auth_events)

    redact_level = _get_named_level(auth_events, "redact", 50)

    if user_level >= redact_level:
        return False

    redacter_domain = get_domain_from_id(event.event_id)
    redactee_domain = get_domain_from_id(event.redacts)
    if redacter_domain == redactee_domain:
        return True

    raise AuthError(
        403,
        "You don't have permission to redact events"
    )


def _check_power_levels(event, auth_events):
    user_list = event.content.get("users", {})
    # Validate users
    for k, v in user_list.items():
        try:
            UserID.from_string(k)
        except:
            raise SynapseError(400, "Not a valid user_id: %s" % (k,))

        try:
            int(v)
        except:
            raise SynapseError(400, "Not a valid power level: %s" % (v,))

    key = (event.type, event.state_key, )
    current_state = auth_events.get(key)

    if not current_state:
        return

    user_level = get_user_power_level(event.user_id, auth_events)

    # Check other levels:
    levels_to_check = [
        ("users_default", None),
        ("events_default", None),
        ("state_default", None),
        ("ban", None),
        ("redact", None),
        ("kick", None),
        ("invite", None),
    ]

    old_list = current_state.content.get("users")
    for user in set(old_list.keys() + user_list.keys()):
        levels_to_check.append(
            (user, "users")
        )

    old_list = current_state.content.get("events")
    new_list = event.content.get("events")
    for ev_id in set(old_list.keys() + new_list.keys()):
        levels_to_check.append(
            (ev_id, "events")
        )

    old_state = current_state.content
    new_state = event.content

    for level_to_check, dir in levels_to_check:
        old_loc = old_state
        new_loc = new_state
        if dir:
            old_loc = old_loc.get(dir, {})
            new_loc = new_loc.get(dir, {})

        if level_to_check in old_loc:
            old_level = int(old_loc[level_to_check])
        else:
            old_level = None

        if level_to_check in new_loc:
            new_level = int(new_loc[level_to_check])
        else:
            new_level = None

        if new_level is not None and old_level is not None:
            if new_level == old_level:
                continue

        if dir == "users" and level_to_check != event.user_id:
            if old_level == user_level:
                raise AuthError(
                    403,
                    "You don't have permission to remove ops level equal "
                    "to your own"
                )

        if old_level > user_level or new_level > user_level:
            raise AuthError(
                403,
                "You don't have permission to add ops level greater "
                "than your own"
            )


def _get_power_level_event(auth_events):
    key = (EventTypes.PowerLevels, "", )
    return auth_events.get(key)


def get_user_power_level(user_id, auth_events):
    power_level_event = _get_power_level_event(auth_events)

    if power_level_event:
        level = power_level_event.content.get("users", {}).get(user_id)
        if not level:
            level = power_level_event.content.get("users_default", 0)

        if level is None:
            return 0
        else:
            return int(level)
    else:
        key = (EventTypes.Create, "", )
        create_event = auth_events.get(key)
        if (create_event is not None and
                create_event.content["creator"] == user_id):
            return 100
        else:
            return 0


def _get_named_level(auth_events, name, default):
    power_level_event = _get_power_level_event(auth_events)

    if not power_level_event:
        return default

    level = power_level_event.content.get(name, None)
    if level is not None:
        return int(level)
    else:
        return default


def _verify_third_party_invite(event, auth_events):
    """
    Validates that the invite event is authorized by a previous third-party invite.

    Checks that the public key, and keyserver, match those in the third party invite,
    and that the invite event has a signature issued using that public key.

    Args:
        event: The m.room.member join event being validated.
        auth_events: All relevant previous context events which may be used
            for authorization decisions.

    Return:
        True if the event fulfills the expectations of a previous third party
        invite event.
    """
    if "third_party_invite" not in event.content:
        return False
    if "signed" not in event.content["third_party_invite"]:
        return False
    signed = event.content["third_party_invite"]["signed"]
    for key in {"mxid", "token"}:
        if key not in signed:
            return False

    token = signed["token"]

    invite_event = auth_events.get(
        (EventTypes.ThirdPartyInvite, token,)
    )
    if not invite_event:
        return False

    if invite_event.sender != event.sender:
        return False

    if event.user_id != invite_event.user_id:
        return False

    if signed["mxid"] != event.state_key:
        return False
    if signed["token"] != token:
        return False

    for public_key_object in get_public_keys(invite_event):
        public_key = public_key_object["public_key"]
        try:
            for server, signature_block in signed["signatures"].items():
                for key_name, encoded_signature in signature_block.items():
                    if not key_name.startswith("ed25519:"):
                        continue
                    verify_key = decode_verify_key_bytes(
                        key_name,
                        decode_base64(public_key)
                    )
                    verify_signed_json(signed, server, verify_key)

                    # We got the public key from the invite, so we know that the
                    # correct server signed the signed bundle.
                    # The caller is responsible for checking that the signing
                    # server has not revoked that public key.
                    return True
        except (KeyError, SignatureVerifyException,):
            continue
    return False


def get_public_keys(invite_event):
    public_keys = []
    if "public_key" in invite_event.content:
        o = {
            "public_key": invite_event.content["public_key"],
        }
        if "key_validity_url" in invite_event.content:
            o["key_validity_url"] = invite_event.content["key_validity_url"]
        public_keys.append(o)
    public_keys.extend(invite_event.content.get("public_keys", []))
    return public_keys


def auth_types_for_event(event):
    """Given an event, return a list of (EventType, StateKey) that may be
    needed to auth the event. The returned list may be a superset of what
    would actually be required depending on the full state of the room.

    Used to limit the number of events to fetch from the database to
    actually auth the event.
    """
    if event.type == EventTypes.Create:
        return []

    auth_types = []

    auth_types.append((EventTypes.PowerLevels, "", ))
    auth_types.append((EventTypes.Member, event.user_id, ))
    auth_types.append((EventTypes.Create, "", ))

    if event.type == EventTypes.Member:
        membership = event.content["membership"]
        if membership in [Membership.JOIN, Membership.INVITE]:
            auth_types.append((EventTypes.JoinRules, "", ))

        auth_types.append((EventTypes.Member, event.state_key, ))

        if membership == Membership.INVITE:
            if "third_party_invite" in event.content:
                key = (
                    EventTypes.ThirdPartyInvite,
                    event.content["third_party_invite"]["signed"]["token"]
                )
                auth_types.append(key)

    return auth_types
