import pytest

import sys, os
sys.path.append(os.path.realpath(os.path.dirname(__file__)+"/.."))

import numpy as np

from deltametrics import mask


def test_mask_init():
    msk = mask.BaseMask(mask_type='test', data=np.zeros((100,100)))
    pass
