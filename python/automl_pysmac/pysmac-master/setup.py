from setuptools import setup, find_packages

setup(
    name = 'pysmac',
    version = '0.9.1',
    packages = find_packages(),
    install_requires = ['docutils>=0.3', 'setuptools', 'numpy', 'pynisher'],
    author = "Stefan Falkner and Tobias Domhan (python wrapper). Frank Hutter, Holger Hoos, Kevin Leyton-Brown, Kevin Murphy and Steve Ramage (SMAC)",
    author_email = "sfalkner@informatik.uni-freiburg.de",
    description = "python interface to the hyperparameter optimization tool SMAC.",
    include_package_data = True,
    keywords = "hyperparameter parameter optimization hyperopt bayesian smac global",
    license = "SMAC is free for academic & non-commercial usage. Please contact Frank Hutter(fh@informatik.uni-freiburg.de) to discuss obtaining a license for commercial purposes.",
    url = "http://www.cs.ubc.ca/labs/beta/Projects/SMAC/",
    download_url = 'https://github.com/sfalkner/pysmac/tarball/0.9',
    extras_require={
        'analyzer': ['matplotlib'],
    },
)
