# coding=utf-8
"""
用于py2exe生成单一exe文件
"""

import py2exe
from distutils.core import setup

options = {
    'py2exe':
    {
        'compressed'	: 1,
        'optimize'		: 2,
        'bundle_files'	: 1,
        'includes'		: ['encodings', 'encodings.*']
    }
}

setup(
    name		= 'SHU-XK',
    version		= '1.0.0',
    description	= 'SHU-XK',
    options		= options,
    zipfile		= None,
    console     = [{'script': 'SHU-XK.py', }]
)
