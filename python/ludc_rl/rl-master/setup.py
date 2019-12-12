#!/usr/bin/env python

#from distutils.core import setup
from setuptools import setup,find_packages

setup(name='torch_rl',
      version='1.0',
      description='A PyTorch-based RL Library',
      author='Ludovic Denoyer',
      author_email='ludovic.denoyer@lip6.fr',
      url='https://github.com/ludc/rl',
      packages=find_packages()
     )
