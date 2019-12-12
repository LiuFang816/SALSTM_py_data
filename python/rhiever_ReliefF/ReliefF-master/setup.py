#!/usr/bin/env python
# -*- coding: utf-8 -*-
from setuptools import setup, find_packages

def calculate_version():
    initpy = open('ReliefF/_version.py').read().split('\n')
    version = list(filter(lambda x: '__version__' in x, initpy))[0].split('\'')[1]
    return version

package_version = calculate_version()

setup(
    name='ReliefF',
    version=package_version,
    author='Randal S. Olson',
    author_email='rso@randalolson.com',
    packages=find_packages(),
    url='https://github.com/rhiever/ReliefF',
    license='License :: OSI Approved :: MIT License',
    description=('ReliefF feature selection algorithms'),
    long_description='''
This package contains implementations of the ReliefF family of feature selection algorithms: https://en.wikipedia.org/wiki/Relief_(feature_selection)

These algorithms excel at identifying features that are predictive of the outcome in supervised learning problems, and are especially good at identifying feature interactions that are normally overlooked by standard feature selection algorithms.

The main benefit of ReliefF algorithms is that they identify feature interactions without having to exhaustively check every pairwise interaction, thus taking significantly less time than exhaustive pairwise search.

ReliefF algorithms are commonly applied to genetic analyses, where epistasis (i.e., feature interactions) is common. However, the algorithms implemented in this package can be applied to any supervised classification data set.

Contact
=============
If you have any questions or comments about ReliefF, please feel free to contact me via:

E-mail: rso@randalolson.com

or Twitter: https://twitter.com/randal_olson

This project is hosted at https://github.com/rhiever/ReliefF
''',
    zip_safe=True,
    install_requires=['numpy', 'scipy', 'scikit-learn'],
    classifiers=[
        'Intended Audience :: Developers',
        'Intended Audience :: Information Technology',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Utilities'
    ],
    keywords=['feature selection', 'feature preprocessing', 'machine learning', 'data science', 'supervised classification', 'feature importance', 'epistasis', 'genetic analysis'],
)
