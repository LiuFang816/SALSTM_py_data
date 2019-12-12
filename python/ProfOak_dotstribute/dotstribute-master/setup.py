#!/usr/bin/env python

from distutils.core import setup
import os

old_name = 'dotstribute.py'
new_name = 'dotstribute'

# you don't want to type .py when executing this, do you?
if os.path.exists(old_name):
    os.rename(old_name, new_name)

setup(name='Dotstribute',
      version='1.0',
      description='Make your dotfiles easier to manage',
      author='ProfOak',
      author_email='OpenProfOak@gmail.com',
      url='https://github.com/ProfOak/dotstribute',
      scripts=[new_name],
      keywords=['dotfile github version control'],
     )

# back to normal :-)
if os.path.exists(new_name):
    os.rename(new_name, old_name)

