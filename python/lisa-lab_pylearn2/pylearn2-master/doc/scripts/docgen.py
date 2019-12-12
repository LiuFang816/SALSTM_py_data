
from __future__ import print_function

from collections import defaultdict
import inspect
import getopt
import os
import shutil
import sys


if __name__ == '__main__':

    throot = "/".join(sys.path[0].split("/")[:-2])

    options = defaultdict(bool)
    options.update(dict([x, y or True] for x, y in getopt.getopt(sys.argv[1:], 'o:', ['epydoc', 'rst', 'help', 'nopdf', 'test'])[0]))
    if options['--help']:
        print('Usage: %s [OPTIONS]' % sys.argv[0])
        print('  -o <dir>: output the html files in the specified dir')
        print('  --rst: only compile the doc (requires sphinx)')
        print('  --nopdf: do not produce a PDF file from the doc, only HTML')
        print('  --help: this help')
        # --test: build the docs with warnings=errors to test them (exclusive)
        sys.exit(0)

    options['--all'] = not options['--rst']

    def mkdir(path):
        try:
            os.mkdir(path)
        except OSError:
            pass

    outdir = options['-o'] or (throot + '/html')
    mkdir(outdir)
    os.chdir(outdir)
    mkdir("doc")

    # Make sure the appropriate 'theano' directory is in the PYTHONPATH
    pythonpath = os.environ.get('PYTHONPATH', '')
    pythonpath = throot + ':' + pythonpath
    os.environ['PYTHONPATH'] = pythonpath

    if options['--test']:
        import sphinx
        sys.path[0:0] = [os.path.join(throot, 'doc')]
        out = sphinx.main(['', '-b' 'text', '-W',
                           '-E', os.path.join(throot, 'doc'), '.'])
        sys.exit(out)
    elif options['--all'] or options['--rst']:
        import sphinx
        sys.path[0:0] = [os.path.join(throot, 'doc')]
        sphinx.main(['', '-E', os.path.join(throot, 'doc'), '.'])

        if not options['--nopdf']:
            # Generate latex file in a temp directory
            import tempfile
            workdir = tempfile.mkdtemp()
            sphinx.main(['', '-E', '-b', 'latex',
                os.path.join(throot, 'doc'), workdir])
            # Compile to PDF
            os.chdir(workdir)
            os.system('make')
            try:
                shutil.copy(os.path.join(workdir, 'pylearn2.pdf'), outdir)
                os.chdir(outdir)
                shutil.rmtree(workdir)
            except OSError as e:
                print('OSError:', e)
            except IOError as e:
                print('IOError:', e)
