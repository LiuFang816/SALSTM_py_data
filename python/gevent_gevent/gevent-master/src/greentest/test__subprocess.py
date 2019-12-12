import sys
import os
import errno
import greentest
import gevent
from gevent import subprocess
import time
import gc
import tempfile


PYPY = hasattr(sys, 'pypy_version_info')
PY3 = sys.version_info[0] >= 3


if subprocess.mswindows:
    SETBINARY = 'import msvcrt; msvcrt.setmode(sys.stdout.fileno(), os.O_BINARY);'
else:
    SETBINARY = ''


python_universal_newlines = hasattr(sys.stdout, 'newlines')
# The stdlib of Python 3 on Windows doesn't properly handle universal newlines
# (it produces broken results compared to Python 2)
# See gevent.subprocess for more details.
if PY3 and subprocess.mswindows:
    python_universal_newlines_broken = True
else:
    python_universal_newlines_broken = False


class Test(greentest.TestCase):

    def setUp(self):
        gc.collect()
        gc.collect()

    def test_exit(self):
        popen = subprocess.Popen([sys.executable, '-c', 'import sys; sys.exit(10)'])
        self.assertEqual(popen.wait(), 10)

    def test_wait(self):
        popen = subprocess.Popen([sys.executable, '-c', 'import sys; sys.exit(11)'])
        gevent.wait([popen])
        self.assertEqual(popen.poll(), 11)

    def test_child_exception(self):
        try:
            subprocess.Popen(['*']).wait()
        except OSError as ex:
            assert ex.errno == 2, ex
        else:
            raise AssertionError('Expected OSError: [Errno 2] No such file or directory')

    def test_leak(self):
        num_before = greentest.get_number_open_files()
        p = subprocess.Popen([sys.executable, "-c", "print()"],
                             stdout=subprocess.PIPE)
        p.wait()
        del p
        if PYPY:
            gc.collect()
            gc.collect()

        num_after = greentest.get_number_open_files()
        self.assertEqual(num_before, num_after)


    def test_communicate(self):
        p = subprocess.Popen([sys.executable, "-c",
                              'import sys,os;'
                              'sys.stderr.write("pineapple");'
                              'sys.stdout.write(sys.stdin.read())'],
                             stdin=subprocess.PIPE,
                             stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE)
        (stdout, stderr) = p.communicate(b"banana")
        self.assertEqual(stdout, b"banana")
        if sys.executable.endswith('-dbg'):
            assert stderr.startswith(b'pineapple')
        else:
            self.assertEqual(stderr, b"pineapple")

    def test_universal1(self):
        p = subprocess.Popen([sys.executable, "-c",
                              'import sys,os;' + SETBINARY +
                              'sys.stdout.write("line1\\n");'
                              'sys.stdout.flush();'
                              'sys.stdout.write("line2\\r");'
                              'sys.stdout.flush();'
                              'sys.stdout.write("line3\\r\\n");'
                              'sys.stdout.flush();'
                              'sys.stdout.write("line4\\r");'
                              'sys.stdout.flush();'
                              'sys.stdout.write("\\nline5");'
                              'sys.stdout.flush();'
                              'sys.stdout.write("\\nline6");'],
                             stdout=subprocess.PIPE,
                             universal_newlines=1,
                             bufsize=1)
        try:
            stdout = p.stdout.read()
            if python_universal_newlines:
                # Interpreter with universal newline support
                if not python_universal_newlines_broken:
                    self.assertEqual(stdout,
                                     "line1\nline2\nline3\nline4\nline5\nline6")
                else:
                    # Note the extra newline after line 3
                    self.assertEqual(stdout,
                                     'line1\nline2\nline3\n\nline4\n\nline5\nline6')
            else:
                # Interpreter without universal newline support
                self.assertEqual(stdout,
                                 "line1\nline2\rline3\r\nline4\r\nline5\nline6")
        finally:
            p.stdout.close()

    def test_universal2(self):
        p = subprocess.Popen([sys.executable, "-c",
                              'import sys,os;' + SETBINARY +
                              'sys.stdout.write("line1\\n");'
                              'sys.stdout.flush();'
                              'sys.stdout.write("line2\\r");'
                              'sys.stdout.flush();'
                              'sys.stdout.write("line3\\r\\n");'
                              'sys.stdout.flush();'
                              'sys.stdout.write("line4\\r\\nline5");'
                              'sys.stdout.flush();'
                              'sys.stdout.write("\\nline6");'],
                             stdout=subprocess.PIPE,
                             universal_newlines=1,
                             bufsize=1)
        try:
            stdout = p.stdout.read()
            if python_universal_newlines:
                # Interpreter with universal newline support
                if not python_universal_newlines_broken:
                    self.assertEqual(stdout,
                                     "line1\nline2\nline3\nline4\nline5\nline6")
                else:
                    # Note the extra newline after line 3
                    self.assertEqual(stdout,
                                     'line1\nline2\nline3\n\nline4\n\nline5\nline6')
            else:
                # Interpreter without universal newline support
                self.assertEqual(stdout,
                                 "line1\nline2\rline3\r\nline4\r\nline5\nline6")
        finally:
            p.stdout.close()

    if sys.platform != 'win32':

        def test_nonblock_removed(self):
            # see issue #134
            r, w = os.pipe()
            stdin = subprocess.FileObject(r)
            p = subprocess.Popen(['grep', 'text'], stdin=stdin)
            try:
                # Closing one half of the pipe causes Python 3 on OS X to terminate the
                # child process; it exits with code 1 and the assert that p.poll is None
                # fails. Removing the close lets it pass under both Python 3 and 2.7.
                # If subprocess.Popen._remove_nonblock_flag is changed to a noop, then
                # the test fails (as expected) even with the close removed
                #os.close(w)
                time.sleep(0.1)
                self.assertEqual(p.poll(), None)
            finally:
                if p.poll() is None:
                    p.kill()
                stdin.close()
                os.close(w)

    def test_issue148(self):
        for i in range(7):
            try:
                subprocess.Popen('this_name_must_not_exist')
            except OSError as ex:
                if ex.errno != errno.ENOENT:
                    raise
            else:
                raise AssertionError('must fail with ENOENT')

    def test_check_output_keyword_error(self):
        try:
            subprocess.check_output([sys.executable, '-c', 'import sys; sys.exit(44)'])
        except subprocess.CalledProcessError as e:
            self.assertEqual(e.returncode, 44)
        else:
            raise AssertionError('must fail with CalledProcessError')

    def test_popen_bufsize(self):
        # Test that subprocess has unbuffered output by default
        # (as the vanilla subprocess module)
        if PY3:
            # The default changed under python 3.
            return
        p = subprocess.Popen([sys.executable, '-u', '-c',
                              'import sys; sys.stdout.write(sys.stdin.readline())'],
                             stdin=subprocess.PIPE, stdout=subprocess.PIPE)
        p.stdin.write(b'foobar\n')
        r = p.stdout.readline()
        self.assertEqual(r, b'foobar\n')

    if sys.platform != 'win32':

        def test_subprocess_in_native_thread(self):
            # gevent.subprocess doesn't work from a background
            # native thread. See #688
            from gevent import monkey

            # must be a native thread; defend against monkey-patching
            ex = []
            Thread = monkey.get_original('threading', 'Thread')

            def fn():
                try:
                    gevent.subprocess.Popen('echo 123', shell=True)
                    raise AssertionError("Should not be able to construct Popen")
                except Exception as e:
                    ex.append(e)

            thread = Thread(target=fn)
            thread.start()
            thread.join()

            self.assertEqual(len(ex), 1)
            self.assertTrue(isinstance(ex[0], TypeError), ex)
            self.assertEqual(ex[0].args[0], 'child watchers are only available on the default loop')

        test_subprocess_in_native_thread.ignore_leakcheck = True

class RunFuncTestCase(greentest.TestCase):
    # Based on code from python 3.6

    __timeout__ = 6

    def run_python(self, code, **kwargs):
        """Run Python code in a subprocess using subprocess.run"""
        argv = [sys.executable, "-c", code]
        return subprocess.run(argv, **kwargs)

    def test_returncode(self):
        # call() function with sequence argument
        cp = self.run_python("import sys; sys.exit(47)")
        self.assertEqual(cp.returncode, 47)
        with self.assertRaises(subprocess.CalledProcessError):
            cp.check_returncode()

    def test_check(self):
        with self.assertRaises(subprocess.CalledProcessError) as c:
            self.run_python("import sys; sys.exit(47)", check=True)
        self.assertEqual(c.exception.returncode, 47)

    def test_check_zero(self):
        # check_returncode shouldn't raise when returncode is zero
        cp = self.run_python("import sys; sys.exit(0)", check=True)
        self.assertEqual(cp.returncode, 0)

    def test_timeout(self):
        # run() function with timeout argument; we want to test that the child
        # process gets killed when the timeout expires.  If the child isn't
        # killed, this call will deadlock since subprocess.run waits for the
        # child.
        with self.assertRaises(subprocess.TimeoutExpired):
            self.run_python("while True: pass", timeout=0.0001)

    def test_capture_stdout(self):
        # capture stdout with zero return code
        cp = self.run_python("print('BDFL')", stdout=subprocess.PIPE)
        self.assertIn(b'BDFL', cp.stdout)

    def test_capture_stderr(self):
        cp = self.run_python("import sys; sys.stderr.write('BDFL')",
                             stderr=subprocess.PIPE)
        self.assertIn(b'BDFL', cp.stderr)

    def test_check_output_stdin_arg(self):
        # run() can be called with stdin set to a file
        with tempfile.TemporaryFile() as tf:
            tf.write(b'pear')
            tf.seek(0)
            cp = self.run_python(
                "import sys; sys.stdout.write(sys.stdin.read().upper())",
                stdin=tf, stdout=subprocess.PIPE)
            self.assertIn(b'PEAR', cp.stdout)

    def test_check_output_input_arg(self):
        # check_output() can be called with input set to a string
        cp = self.run_python(
            "import sys; sys.stdout.write(sys.stdin.read().upper())",
            input=b'pear', stdout=subprocess.PIPE)
        self.assertIn(b'PEAR', cp.stdout)

    def test_check_output_stdin_with_input_arg(self):
        # run() refuses to accept 'stdin' with 'input'
        with tempfile.TemporaryFile() as tf:
            tf.write(b'pear')
            tf.seek(0)
            with self.assertRaises(ValueError,
                                   msg="Expected ValueError when stdin and input args supplied.") as c:
                self.run_python("print('will not be run')",
                                stdin=tf, input=b'hare')
            self.assertIn('stdin', c.exception.args[0])
            self.assertIn('input', c.exception.args[0])

    def test_check_output_timeout(self):
        with self.assertRaises(subprocess.TimeoutExpired) as c:
            self.run_python(
                (
                    "import sys, time\n"
                    "sys.stdout.write('BDFL')\n"
                    "sys.stdout.flush()\n"
                    "time.sleep(3600)"
                ),
                # Some heavily loaded buildbots (sparc Debian 3.x) require
                # this much time to start and print.
                timeout=3, stdout=subprocess.PIPE)
        self.assertEqual(c.exception.output, b'BDFL')
        # output is aliased to stdout
        self.assertEqual(c.exception.stdout, b'BDFL')

    def test_run_kwargs(self):
        newenv = os.environ.copy()
        newenv["FRUIT"] = "banana"
        cp = self.run_python(('import sys, os;'
                              'sys.exit(33 if os.getenv("FRUIT")=="banana" else 31)'),
                             env=newenv)
        self.assertEqual(cp.returncode, 33)


if __name__ == '__main__':
    greentest.main()
