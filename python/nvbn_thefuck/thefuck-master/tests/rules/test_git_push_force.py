import pytest
from thefuck.rules.git_push_force import match, get_new_command
from tests.utils import Command


git_err = '''
To /tmp/foo
 ! [rejected]        master -> master (non-fast-forward)
 error: failed to push some refs to '/tmp/bar'
 hint: Updates were rejected because the tip of your current branch is behind
 hint: its remote counterpart. Integrate the remote changes (e.g.
 hint: 'git pull ...') before pushing again.
 hint: See the 'Note about fast-forwards' in 'git push --help' for details.
'''

git_uptodate = 'Everything up-to-date'
git_ok = '''
Counting objects: 3, done.
Delta compression using up to 4 threads.
Compressing objects: 100% (2/2), done.
Writing objects: 100% (3/3), 282 bytes | 0 bytes/s, done.
Total 3 (delta 0), reused 0 (delta 0)
To /tmp/bar
   514eed3..f269c79  master -> master
'''


@pytest.mark.parametrize('command', [
    Command(script='git push', stderr=git_err),
    Command(script='git push nvbn', stderr=git_err),
    Command(script='git push nvbn master', stderr=git_err)])
def test_match(command):
    assert match(command)


@pytest.mark.parametrize('command', [
    Command(script='git push', stderr=git_ok),
    Command(script='git push', stderr=git_uptodate),
    Command(script='git push nvbn', stderr=git_ok),
    Command(script='git push nvbn master', stderr=git_uptodate),
    Command(script='git push nvbn', stderr=git_ok),
    Command(script='git push nvbn master', stderr=git_uptodate)])
def test_not_match(command):
    assert not match(command)


@pytest.mark.parametrize('command, output', [
    (Command(script='git push', stderr=git_err), 'git push --force-with-lease'),
    (Command(script='git push nvbn', stderr=git_err), 'git push --force-with-lease nvbn'),
    (Command(script='git push nvbn master', stderr=git_err), 'git push --force-with-lease nvbn master')])
def test_get_new_command(command, output):
    assert get_new_command(command) == output
