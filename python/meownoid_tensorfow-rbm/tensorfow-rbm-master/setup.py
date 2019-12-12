from distutils.core import setup

desc = """\
tensorflow-rbm
==============
Tensorflow implementation of Restricted Boltzman Machine
for layerwise pretraining of deep autoencoders
"""

setup(name='tfrbm',
      version='0.0.2',
      author='Egor Malykh, Michal Lukac',
      author_email='fnk@fea.st',
      long_description=desc,
      packages=['tfrbm'],
      url='https://github.com/meownoid/tensorflow-rbm')
