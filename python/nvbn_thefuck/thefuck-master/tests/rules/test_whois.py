import pytest
from thefuck.rules.whois import match, get_new_command
from tests.utils import Command


@pytest.mark.parametrize('command', [
    Command(script='whois https://en.wikipedia.org/wiki/Main_Page'),
    Command(script='whois https://en.wikipedia.org/'),
    Command(script='whois meta.unix.stackexchange.com')])
def test_match(command):
    assert match(command)


def test_not_match():
    assert not match(Command(script='whois'))


# `whois com` actually makes sense
@pytest.mark.parametrize('command, new_command', [
    (Command('whois https://en.wikipedia.org/wiki/Main_Page'), 'whois en.wikipedia.org'),
    (Command('whois https://en.wikipedia.org/'), 'whois en.wikipedia.org'),
    (Command('whois meta.unix.stackexchange.com'), ['whois unix.stackexchange.com',
                                                    'whois stackexchange.com',
                                                    'whois com'])])
def test_get_new_command(command, new_command):
    assert get_new_command(command) == new_command
