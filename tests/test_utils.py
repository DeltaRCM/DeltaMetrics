import pytest

import sys
import os

import numpy as np
import matplotlib.pyplot as plt

from deltametrics import utils
from deltametrics import mobility as mob


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


chmap = np.zeros((5, 4, 4))
# define time = 0
chmap[0, :, 1] = 1
# every time step one cell of the channel will migrate one pixel to the right
for i in range(1, 5):
    chmap[i, :, :] = chmap[i-1, :, :].copy()
    chmap[i, -1*i, 1] = 0
    chmap[i, -1*i, 2] = 1
# define the fluvial surface - entire 4x4 area
fsurf = np.ones((4, 4))
# define the index corresponding to the basemap at time 0
basevalue = [0]
# define the size of the time window to use
time_window = 5


def test_linear_fit():
    """Test linear curve fitting."""
    ch_abandon = mob.calculate_channel_abandonment(chmap, basevalue,
                                                   time_window)
    yfit, cov, err = utils.curve_fit(ch_abandon, fit='linear')
    assert pytest.approx(yfit == np.array([4.76315477e-24, 2.50000000e-01,
                                           5.00000000e-01, 7.50000000e-01,
                                           1.00000000e+00]))
    assert pytest.approx(cov == np.array([[1.76300984e-25, -0.00000000e+00],
                                          [0.00000000e+00,  5.28902953e-24]]))
    assert pytest.approx(err == np.array([4.19882108e-13, 2.29978902e-12]))


def test_harmonic_fit():
    """Test harmonic curve fitting."""
    ch_abandon = mob.calculate_channel_abandonment(chmap, basevalue,
                                                   time_window)
    yfit, cov, err = utils.curve_fit(ch_abandon, fit='harmonic')
    assert pytest.approx(yfit == np.array([-0.25986438, 0.41294455,
                                           0.11505591, 0.06683947,
                                           0.04710091]))
    assert pytest.approx(cov == np.array([[0.50676407, 1.26155952],
                                          [1.26155952, 4.3523343]]))
    assert pytest.approx(err == np.array([0.71187364, 2.08622489]))


def test_invalid_fit():
    """Test invalid fit parameter."""
    ch_abandon = mob.calculate_channel_abandonment(chmap, basevalue,
                                                   time_window)
    with pytest.raises(ValueError):
        utils.curve_fit(ch_abandon, fit='invalid')


def test_exponential_fit():
    """Test exponential fitting."""
    ydata = np.array([10, 5, 2, 1])
    yfit, cov, err = utils.curve_fit(ydata, fit='exponential')
    assert pytest.approx(yfit == np.array([10.02900253, 4.85696353,
                                           2.22612537, 0.88790858]))
    assert pytest.approx(cov == np.array([[0.0841566, 0.04554967, 0.01139969],
                                          [0.04554967, 0.59895713, 0.08422946],
                                          [0.01139969, 0.08422946,
                                           0.01327807]]))
    assert pytest.approx(err == np.array([0.29009757, 0.77392321, 0.11523053]))


def test_format_number_float():
    _val = float(5.2)
    _fnum = utils.format_number(_val)
    assert _fnum == '10'

    _val = float(50.2)
    _fnum = utils.format_number(_val)
    assert _fnum == '50'

    _val = float(15.0)
    _fnum = utils.format_number(_val)
    assert _fnum == '20'


def test_format_number_int():
    _val = int(5)
    _fnum = utils.format_number(_val)
    assert _fnum == '0'

    _val = int(6)
    _fnum = utils.format_number(_val)
    assert _fnum == '10'

    _val = int(52)
    _fnum = utils.format_number(_val)
    assert _fnum == '50'

    _val = int(15)
    _fnum = utils.format_number(_val)
    assert _fnum == '20'


def test_format_table_float():
    _val = float(5.2)
    _fnum = utils.format_table(_val)
    assert _fnum == '5.2'

    _val = float(50.2)
    _fnum = utils.format_table(_val)
    assert _fnum == '50.2'

    _val = float(15.0)
    _fnum = utils.format_table(_val)
    assert _fnum == '15.0'

    _val = float(15.03689)
    _fnum = utils.format_table(_val)
    assert _fnum == '15.0'

    _val = float(15.0689)
    _fnum = utils.format_table(_val)
    assert _fnum == '15.1'


def test_format_table_float():
    _val = int(5)
    _fnum = utils.format_table(_val)
    assert _fnum == '5'

    _val = int(5.8)
    _fnum = utils.format_table(_val)
    assert _fnum == '5'

    _val = int(5.2)
    _fnum = utils.format_table(_val)
    assert _fnum == '5'
