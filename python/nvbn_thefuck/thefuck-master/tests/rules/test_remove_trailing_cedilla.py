import pytest
from thefuck.rules.remove_trailing_cedilla import match, get_new_command, CEDILLA
from tests.utils import Command


@pytest.mark.parametrize('command', [
    Command(script='wrong' + CEDILLA),
    Command(script='wrong with args' + CEDILLA)])
def test_match(command):
    assert match(command)


@pytest.mark.parametrize('command, new_command', [
    (Command('wrong' + CEDILLA), 'wrong'),
    (Command('wrong with args' + CEDILLA), 'wrong with args')])
def test_get_new_command(command, new_command):
    assert get_new_command(command) == new_command
