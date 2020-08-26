import pytest

import sys
import os

import numpy as np
import matplotlib.pyplot as plt

from deltametrics import utils


class TestNoStratigraphyError:

    def test_needs_obj_argument(self):
        with pytest.raises(TypeError):
            raise utils.NoStratigraphyError()

    def test_only_obj_argument(self):
        _mtch = "'str' object has no*."
        with pytest.raises(utils.NoStratigraphyError, match=_mtch):
            raise utils.NoStratigraphyError('someobj')

    def test_obj_and_var(self):
        _mtch = "'str' object has no attribute 'somevar'."
        with pytest.raises(utils.NoStratigraphyError, match=_mtch):
            raise utils.NoStratigraphyError('someobj', 'somevar')
