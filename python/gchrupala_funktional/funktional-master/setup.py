# encoding: utf-8
from setuptools import setup

setup(name='funktional',
      version='0.5',
      description='A minimalistic toolkit for functionally composable neural network layers with Theano.',
      url='https://github.com/gchrupala/funktional',
      author='Grzegorz Chrupała',
      author_email='g.chrupala@uvt.nl',
      license='MIT',
      packages=['funktional'],
      scripts=['autoencoder.py'],
      install_requires=['Theano'],
      zip_safe=False)
