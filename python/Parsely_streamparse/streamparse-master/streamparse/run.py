"""
Helper script for starting up bolts and spouts.
"""

import argparse
import importlib
import os
import sys

from pystorm.component import _SERIALIZERS


def main():
    """main entry point for Python bolts and spouts"""
    parser = argparse.ArgumentParser(description='Run a bolt/spout class',
                                     epilog='This is internal to streamparse '
                                            'and is used to run spout and bolt '
                                            'classes via ``python -m '
                                            'streamparse.run <class name>``.')
    parser.add_argument('target_class', help='The bolt/spout class to start.')
    parser.add_argument('-s', '--serializer',
                        help='The serialization protocol to use to talk to '
                             'Storm.',
                        choices=_SERIALIZERS.keys(),
                        default='json')
    # Storm sends everything as one string, which is not great
    if len(sys.argv) == 2:
        sys.argv = [sys.argv[0]] + sys.argv[1].split()
    args = parser.parse_args()
    # Add current directory to sys.path so imports will work
    sys.path.append(os.getcwd())
    # Import module
    mod_name, cls_name = args.target_class.rsplit('.', 1)
    mod = importlib.import_module(mod_name)
    # Get class from module and run it
    cls = getattr(mod, cls_name)
    cls(serializer=args.serializer).run()


if __name__ == '__main__':
    main()
