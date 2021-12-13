import os
import shutil

import numpy as np


def sigmoid(x, alpha=1, beta=0):
    """Compute sigmoid of x given midpoint and
    steepness
    """
    p = 1 / (1 + np.exp(-alpha*(x - beta)))
    return p


def create_dir(dir_name):
    """Create a directory given a directory location. If the directory already
    exists it will be removed.
    """
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)
    else:
        shutil.rmtree(dir_name)
        os.mkdir(dir_name)
    return None
