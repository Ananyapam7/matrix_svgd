""" Utilities """

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Logging
# =======

import logging
from colorlog import ColoredFormatter
from sklearn import utils as skutils
from numpy.random import RandomState
import numpy as np
import torch

# Set PyTorch random seed for reproducibility
torch.manual_seed(123)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(123)

# NumPy random seed
seed = 123
np_rng = RandomState(seed)

ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)

formatter = ColoredFormatter(
    "%(log_color)s[%(asctime)s] %(message)s",
#    datefmt='%H:%M:%S.%f',
    datefmt=None,
    reset=True,
    log_colors={
        'DEBUG':    'cyan',
        'INFO':     'white,bold',
        'INFOV':    'cyan,bold',
        'WARNING':  'yellow',
        'ERROR':    'red,bold',
        'CRITICAL': 'red,bg_white',
    },
    secondary_log_colors={},
    style='%'
)
ch.setFormatter(formatter)

log = logging.getLogger('rn')
log.setLevel(logging.DEBUG)
log.handlers = []       # No duplicated handlers
log.propagate = False   # workaround for duplicated logs in ipython
log.addHandler(ch)

logging.addLevelName(logging.INFO + 1, 'INFOV')
def _infov(self, msg, *args, **kwargs):
    self.log(logging.INFO + 1, msg, *args, **kwargs)

logging.Logger.infov = _infov

# PyTorch utility functions
def to_tensor(x, dtype=torch.float32):
    """Convert numpy array to PyTorch tensor"""
    if isinstance(x, torch.Tensor):
        return x
    return torch.tensor(x, dtype=dtype)

def to_numpy(x):
    """Convert PyTorch tensor to numpy array"""
    if isinstance(x, np.ndarray):
        return x
    return x.detach().cpu().numpy()

def set_device(device=None):
    """Set the device for PyTorch operations"""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return device



