import pytest
from thefuck.rules.git_push_pull import match, get_new_command
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

git_err2 = '''
To /tmp/foo
 ! [rejected]        master -> master (non-fast-forward)
 error: failed to push some refs to '/tmp/bar'
hint: Updates were rejected because the remote contains work that you do
hint: not have locally. This is usually caused by another repository pushing
hint: to the same ref. You may want to first integrate the remote changes
hint: (e.g., 'git pull ...') before pushing again.
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
    Command(script='git push', stderr=git_err2),
    Command(script='git push nvbn', stderr=git_err2),
    Command(script='git push nvbn master', stderr=git_err2)])
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
    (Command(script='git push', stderr=git_err), 'git pull && git push'),
    (Command(script='git push nvbn', stderr=git_err),
     'git pull nvbn && git push nvbn'),
    (Command(script='git push nvbn master', stderr=git_err),
     'git pull nvbn master && git push nvbn master')])
def test_get_new_command(command, output):
    assert get_new_command(command) == output


@pytest.mark.parametrize('command, output', [
    (Command(script='git push', stderr=git_err2), 'git pull && git push'),
    (Command(script='git push nvbn', stderr=git_err2),
     'git pull nvbn && git push nvbn'),
    (Command(script='git push nvbn master', stderr=git_err2),
     'git pull nvbn master && git push nvbn master')])
def test_get_new_command(command, output):
    assert get_new_command(command) == output
