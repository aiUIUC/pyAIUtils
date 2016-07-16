# Code to help important module code without installing.
# From: http://docs.python-guide.org/en/latest/writing/structure/#structure-of-code-is-key

import os
import sys
sys.path.insert(0, os.path.abspath('.'))

from aiutils.tftools import layers
from aiutils.tftools import images
from aiutils.data import batch_creators
from aiutils.tftools import placeholder_management
from aiutils.tftools import batch_normalizer
from aiutils.tftools import var_collect
