# pylint: disable=wildcard-import
__version__ = "3.0"

from .blocking import *
from .distributions import *
from .external import *
from .glm import *
from . import gp
from .math import logsumexp, logit, invlogit
from .model import *
from .stats import *
from .sampling import *
from .step_methods import *
from .theanof import *
from .tuning import *
from .variational import *
from .vartypes import *
from . import sampling

from .debug import *

from .diagnostics import *
from .backends.tracetab import *

from .plots import *

from .tests import test

from .data import *

import logging
_log = logging.getLogger('pymc3')
if not logging.root.handlers:
    _log.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    _log.addHandler(handler)
