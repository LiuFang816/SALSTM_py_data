import pytest
from mock import patch
from thefuck.rules import pacman_not_found
from thefuck.rules.pacman_not_found import match, get_new_command
from tests.utils import Command

PKGFILE_OUTPUT_LLC = '''extra/llvm 3.6.0-5      /usr/bin/llc
extra/llvm35 3.5.2-13/usr/bin/llc'''


@pytest.mark.skipif(not getattr(pacman_not_found, 'enabled_by_default', True),
                    reason='Skip if pacman is not available')
@pytest.mark.parametrize('command', [
    Command(script='yaourt -S llc', stderr='error: target not found: llc'),
    Command(script='pacman llc', stderr='error: target not found: llc'),
    Command(script='sudo pacman llc', stderr='error: target not found: llc')])
def test_match(command):
    assert match(command)


@pytest.mark.parametrize('command', [
    Command(script='yaourt -S llc', stderr='error: target not found: llc'),
    Command(script='pacman llc', stderr='error: target not found: llc'),
    Command(script='sudo pacman llc', stderr='error: target not found: llc')])
@patch('thefuck.specific.archlinux.subprocess')
def test_match_mocked(subp_mock, command):
    subp_mock.check_output.return_value = PKGFILE_OUTPUT_LLC
    assert match(command)


@pytest.mark.skipif(not getattr(pacman_not_found, 'enabled_by_default', True),
                    reason='Skip if pacman is not available')
@pytest.mark.parametrize('command, fixed', [
    (Command(script='yaourt -S llc', stderr='error: target not found: llc'), ['yaourt -S extra/llvm', 'yaourt -S extra/llvm35']),
    (Command(script='pacman -S llc', stderr='error: target not found: llc'), ['pacman -S extra/llvm', 'pacman -S extra/llvm35']),
    (Command(script='sudo pacman -S llc', stderr='error: target not found: llc'), ['sudo pacman -S extra/llvm', 'sudo pacman -S extra/llvm35'])])
def test_get_new_command(command, fixed):
    assert get_new_command(command) == fixed


@pytest.mark.parametrize('command, fixed', [
    (Command(script='yaourt -S llc', stderr='error: target not found: llc'), ['yaourt -S extra/llvm', 'yaourt -S extra/llvm35']),
    (Command(script='pacman -S llc', stderr='error: target not found: llc'), ['pacman -S extra/llvm', 'pacman -S extra/llvm35']),
    (Command(script='sudo pacman -S llc', stderr='error: target not found: llc'), ['sudo pacman -S extra/llvm', 'sudo pacman -S extra/llvm35'])])
@patch('thefuck.specific.archlinux.subprocess')
def test_get_new_command_mocked(subp_mock, command, fixed):
    subp_mock.check_output.return_value = PKGFILE_OUTPUT_LLC
    assert get_new_command(command) == fixed
