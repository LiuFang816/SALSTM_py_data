import pytest
from thefuck.rules.python_execute import match, get_new_command
from tests.utils import Command


@pytest.mark.parametrize('command', [
    Command(script='python foo'),
    Command(script='python bar')])
def test_match(command):
    assert match(command)


@pytest.mark.parametrize('command, new_command', [
    (Command('python foo'), 'python foo.py'),
    (Command('python bar'), 'python bar.py')])
def test_get_new_command(command, new_command):
    assert get_new_command(command) == new_command
