#!/usr/bin/env python
from setuptools import setup
import sys

py_entry = 'py%s = pythonpy.__main__:main'
pycompleter_entry = 'pycompleter%s = pythonpy.pycompleter:main'
endings = ('', sys.version[:1], sys.version[:3])
entry_points_scripts = []
for e in endings:
    entry_points_scripts.append(py_entry % e)
    entry_points_scripts.append(pycompleter_entry % e)

setup(
    name='pythonpy',
    version='0.4.11',
    description='python -c, with tab completion and shorthand',
    #data_files=data_files,
    license='MIT',
    url='https://github.com/Russell91/pythonpy',
    long_description='https://github.com/Russell91/pythonpy',
    packages=['pythonpy', 'pythonpy.completion'],
    package_data={'pythonpy': ['completion/pycompletion.sh']},
    scripts=['pythonpy/find_pycompletion.sh'],
    entry_points = {
        'console_scripts': entry_points_scripts
    },
)
