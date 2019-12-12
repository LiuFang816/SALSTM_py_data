# coding: utf-8
from __future__ import unicode_literals

from ...language import Language
from ...attrs import LANG
from ...fr.language_data import get_tokenizer_exceptions, STOP_WORDS
from ...language_data.punctuation import TOKENIZER_INFIXES, ALPHA

import pytest


@pytest.fixture
def fr_tokenizer_w_infix():
    SPLIT_INFIX = r'(?<=[{a}]\')(?=[{a}])'.format(a=ALPHA)

    # create new Language subclass to add to default infixes
    class French(Language):
        lang = 'fr'

        class Defaults(Language.Defaults):
            lex_attr_getters = dict(Language.Defaults.lex_attr_getters)
            lex_attr_getters[LANG] = lambda text: 'fr'
            tokenizer_exceptions = get_tokenizer_exceptions()
            stop_words = STOP_WORDS
            infixes = TOKENIZER_INFIXES + [SPLIT_INFIX]

    return French.Defaults.create_tokenizer()


@pytest.mark.parametrize('text,expected_tokens', [("l'avion", ["l'", "avion"]),
                                                  ("j'ai", ["j'", "ai"])])
def test_issue768(fr_tokenizer_w_infix, text, expected_tokens):
    """Allow zero-width 'infix' token during the tokenization process."""
    tokens = fr_tokenizer_w_infix(text)
    assert len(tokens) == 2
    assert [t.text for t in tokens] == expected_tokens
