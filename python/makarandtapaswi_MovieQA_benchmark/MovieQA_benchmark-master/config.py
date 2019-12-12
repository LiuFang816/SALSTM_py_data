"""Define file paths."""

import os


PACKAGE_DIRECTORY = os.path.dirname(os.path.abspath(__file__))

MOVIES_JSON = os.path.join(PACKAGE_DIRECTORY, 'data/movies.json')
QA_JSON = os.path.join(PACKAGE_DIRECTORY, 'data/qa.json')
SPLIT_JSON = os.path.join(PACKAGE_DIRECTORY, 'data/splits.json')

