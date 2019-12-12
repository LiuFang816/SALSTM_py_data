# encoding: utf8
from __future__ import unicode_literals

from .. import language_data as base
from ..language_data import update_exc, strings_to_exc
from ..symbols import ORTH, LEMMA

from .stop_words import STOP_WORDS
from .tokenizer_exceptions import TOKENIZER_EXCEPTIONS, ORTH_ONLY


def get_time_exc(hours):
    exc = {
        "12m.": [
            {ORTH: "12"},
            {ORTH: "m.", LEMMA: "p.m."}
        ]
    }

    for hour in hours:
        exc["%sa.m." % hour] = [
            {ORTH: hour},
            {ORTH: "a.m."}
        ]

        exc["%sp.m." % hour] = [
            {ORTH: hour},
            {ORTH: "p.m."}
        ]

        exc["%sam" % hour] = [
            {ORTH: hour},
            {ORTH: "am", LEMMA: "a.m."}
        ]

        exc["%spm" % hour] = [
            {ORTH: hour},
            {ORTH: "pm", LEMMA: "p.m."}
        ]
    return exc


STOP_WORDS = set(STOP_WORDS)


TOKENIZER_EXCEPTIONS = dict(TOKENIZER_EXCEPTIONS)
update_exc(TOKENIZER_EXCEPTIONS, strings_to_exc(ORTH_ONLY))
update_exc(TOKENIZER_EXCEPTIONS, get_time_exc(
    ['%d' % hour for hour in range(1, 12 + 1)]))
update_exc(TOKENIZER_EXCEPTIONS, strings_to_exc(base.EMOTICONS))
update_exc(TOKENIZER_EXCEPTIONS, strings_to_exc(base.ABBREVIATIONS))


__all__ = ["TOKENIZER_EXCEPTIONS", "STOP_WORDS"]
