
import sys
import os
import os.path
import datetime
import platform

try:
    # use setuptools if available
    from setuptools import setup, Extension
except ImportError:
    from distutils.core import setup, Extension

VERSION = '1.3.7'

extra_setup_args = {}


# support 'test' target if setuptools/distribute is available

if 'setuptools' in sys.modules:
    extra_setup_args['test_suite'] = 'lutorpy.tests.suite'
    extra_setup_args["zip_safe"] = False

torch_install_dir = os.getenv('TORCH_INSTALL')
if torch_install_dir is None:
    default_torch_path = os.path.join(os.path.expanduser("~"), "torch/install/bin/torch-activate")
    if os.path.exists(default_torch_path):
        torch_install_dir = os.path.join(os.path.expanduser("~"), "torch/install")
    else:
        luajit_path = os.popen("which luajit").read()
        assert 'torch' in luajit_path, 'failed to find torch luajit, please set env variable TORCH_INSTALL to torch/install'
        torch_install_dir = os.path.abspath(os.path.join(luajit_path.strip(),'../../'))
assert torch_install_dir, 'no torch installation is found, please set env variable TORCH_INSTALL to torch/install'

osfamily = platform.uname()[0]
print('torch_install:', torch_install_dir)
print('os family', osfamily)

compile_options = []
if osfamily == 'Windows':
    compile_options.append('/EHsc')

if osfamily in ['Linux', 'Darwin']:
    compile_options.append('-std=c++0x')
    # compile_options.append('-g')
    compile_options.append('-Wno-unused-function')
    compile_options.append('-Wno-unreachable-code')
    compile_options.append('-Wno-strict-prototypes')
    if 'DEBUG' in os.environ:
        compile_options.append('-O0')
        compile_options.append('-g')

runtime_library_dirs = []
libraries = []
extra_link_args = []
#libraries.append('lua5.1')
libraries.append('luaT')
libraries.append('TH')

library_dirs = []
# library_dirs.append('cbuild')
library_dirs.append(torch_install_dir + '/lib')

if osfamily != 'Windows':
    runtime_library_dirs = [torch_install_dir + '/lib']

if osfamily == 'Windows':
    libraries.append('winmm')

if osfamily == 'Darwin':  # Mac OS X
    extra_link_args.append('-Wl,-rpath,' + torch_install_dir + '/lib')

def has_option(name):
    if name in sys.argv[1:]:
        sys.argv.remove(name)
        return True
    return False

import numpy

includes = [ os.path.join(torch_install_dir, 'include'), os.path.join(torch_install_dir, 'include/TH'), numpy.get_include()]

ext_args = {
    'include_dirs': includes,
    'libraries': libraries,
    'library_dirs': library_dirs,
    'extra_link_args': extra_link_args,
    'extra_compile_args': compile_options,
    'runtime_library_dirs':  runtime_library_dirs
}

macros = [('LUA_COMPAT_ALL', None)]
if has_option('--without-assert'):
    macros.append(('CYTHON_WITHOUT_ASSERTIONS', None))
if has_option('--with-lua-checks'):
    macros.append(('LUA_USE_APICHECK', None))
ext_args['define_macros'] = macros


# check if Cython is installed, and use it if requested or necessary
use_cython = has_option('--with-cython')
if not use_cython:
    if not os.path.exists(os.path.join(os.path.dirname(__file__), 'lutorpy', '_lupa.c')):
        print("generated sources not available, need Cython to build")
        use_cython = True
        
cythonize = None
source_extension = ".c"
if use_cython:
    try:
        import Cython.Compiler.Version
        from Cython.Build import cythonize
        print("building with Cython " + Cython.Compiler.Version.version)
        source_extension = ".pyx"
    except ImportError:
        print("WARNING: trying to build with Cython, but it is not installed")
else:
    print("building without Cython")

ext_modules = [
    Extension(
        'lutorpy._lupa',
        sources = ['lutorpy/_lupa'+source_extension],
        **ext_args
    )
]

if cythonize is not None:
    ext_modules = cythonize(ext_modules)

basedir = os.path.abspath(os.path.dirname(__file__))

def read_file(filename):
    with open(os.path.join(basedir, filename)) as f:
        return f.read()


def write_file(filename, content):
    with open(os.path.join(basedir, filename), 'w') as f:
        f.write(content)


long_description = '\n\n'

print('torch install:'+ torch_install_dir)
write_file(os.path.join('lutorpy', 'version.py'), "__version__ = '%s'\n" % VERSION)
write_file(os.path.join('lutorpy', 'torch_path.py'), "__torch_path__ = '%s'\n" % torch_install_dir)
# call distutils

extra_setup_args['package_data'] = {'': ['README.md']}


setup(
    name="lutorpy",
    version=VERSION,
    author="Wei OUYANG",
    author_email="wei.ouyang@cri-paris.org",
    maintainer="Wei OUYANG",
    maintainer_email="wei.ouyang@cri-paris.org",
    url="https://github.com/oeway/lutorpy",
    description="Python wrapper for torch and Lua/LuaJIT",

    long_description=long_description,
    license='MIT style',
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Developers',
        'Intended Audience :: Information Technology',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Cython',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.6',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.2',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Other Scripting Engines',
        'Operating System :: OS Independent',
        'Topic :: Software Development',
    ],
    install_requires=['numpy','future'],
    packages=['lutorpy'],
    ext_modules=ext_modules,
    **extra_setup_args
)
