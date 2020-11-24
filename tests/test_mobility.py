"""Tests for mobility.py."""
import pytest
import sys
import os
import numpy as np

import deltametrics as dm
from deltametrics import cube
from deltametrics import mobility as mob

# initialize a cube directly from path, rather than using sample_data.py
rcm8_path = os.path.join(os.path.dirname(__file__), '..', 'deltametrics',
                         'sample_data', 'files', 'pyDeltaRCM_Output_8.nc')
rcm8cube = cube.DataCube(rcm8_path)
# define some masks once up top
chmask = dm.mask.ChannelMask(rcm8cube['velocity'][20:23, :, :],
                             rcm8cube['eta'][20:23, :, :])
landmask = dm.mask.LandMask(rcm8cube['eta'][20:23, :, :])


def test_check_input_mask():
    """Test that a deltametrics.mask.BaseMask type can be used."""
    # call checker function
    chmap, landmap, basevalues, time_window = mob.check_inputs(chmask,
                                                               [0], 1,
                                                               landmask)
    # assert types
    assert isinstance(chmap, np.ndarray) is True
    assert isinstance(landmap, np.ndarray) is True
    assert isinstance(basevalues, list) is True
    assert isinstance(time_window, int) is True


@pytest.mark.xfail()
def test_check_xarrays():
    """Test that an xarray.DataArray can be used as an input."""
    # call checker function
    chmap, landmap, basevalues, time_window = mob.check_inputs(ch_xarr,
                                                               [0], 1,
                                                               land_xarr)
    # assert types
    assert isinstance(chmap, np.ndarray) is True
    assert isinstance(landmap, np.ndarray) is True
    assert isinstance(basevalues, list) is True
    assert isinstance(time_window, int) is True


def test_check_input_nolandmask():
    """Test that the check input can run without a landmap."""
    # call checker function
    chmap, landmap, basevalues, time_window = mob.check_inputs(chmask,
                                                               [0], 1)
    # assert types
    assert isinstance(chmap, np.ndarray) is True
    assert landmap is None
    assert isinstance(basevalues, list) is True
    assert isinstance(time_window, int) is True


def test_check_input_notbinary_chmap():
    """Test that nonbinary channel map raises error."""
    ch_nonbin = np.zeros((3, 3, 3))
    ch_nonbin[0, 1, 1] = 1
    ch_nonbin[0, 1, 2] = 2
    with pytest.raises(ValueError):
        mob.check_inputs(ch_nonbin, [0], 1)


def test_check_input_notbinary_landmap():
    """Test that nonbinary land map raises error."""
    land_nonbin = np.zeros((3, 3, 3))
    land_nonbin[0, 1, 1] = 1
    land_nonbin[0, 1, 2] = 2
    ch_bin = np.zeros_like(land_nonbin)
    with pytest.raises(ValueError):
        mob.check_inputs(ch_bin, [0], 1, land_nonbin)


def test_check_input_invalid_chmap():
    """Test that an invalid channel map input will throw an error."""
    with pytest.raises(TypeError):
        mob.check_inputs('invalid', [0], 1, landmask)


def test_check_input_invalid_landmap():
    """Test that an invalid landmap will throw an error."""
    with pytest.raises(TypeError):
        mob.check_inputs(chmask, [0], 1, 'invalid')


def test_check_inputs_basevalues_nonlist():
    """Test that a non-list input as basevalues throws an error."""
    with pytest.raises(TypeError):
        mob.check_inputs(chmask, 0, 1)


def test_check_input_invalid_time_window():
    """Test that a non-valid time_window throws an error."""
    with pytest.raises(TypeError):
        mob.check_inputs(chmask, [0], 'invalid')


def test_check_input_2dchanmask():
    """Test that an unexpected channel mask shape throws an error."""
    with pytest.raises(ValueError):
        mob.check_inputs(np.ones((5,)), [0], 1)


def test_check_input_diff_shapes():
    """Test that differently shaped channel and land masks throw an error."""
    with pytest.raises(ValueError):
        mob.check_inputs(chmask, [0], 1, np.ones((3, 3, 3)))


def test_check_input_1dlandmask():
    """Test a 1d landmask that will throw an error."""
    with pytest.raises(ValueError):
        mob.check_inputs(chmask, [0], 1, np.ones((10, 1)))


def test_check_input_exceedmaxvals():
    """Test a basevalue + time window combo that exceeds time indices."""
    with pytest.raises(ValueError):
        mob.check_inputs(chmask, [0], 100)


def test_check_input_castlandmap():
    """Test ability to case a 2D landmask to match 3D channelmap."""
    chmap, landmap, bv, tw = mob.check_inputs(chmask, [0], 1,
                                              landmask.mask[0, :, :])
    assert np.shape(chmap) == np.shape(landmap)


# Test actual mobility functions using a simple example
# domain is a small 4x4 region with a single 4 cell channel
# every timestep, a cell from the channel moves over a column
# colab notebook: https://colab.research.google.com/drive/1-lkswSZSGRkhsArLm245WMSiKvj03YyU?usp=sharing

# first define the domain and channel maps etc.
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


def test_dry_decay():
    """Test dry fraction decay."""
    dryfrac = mob.calculate_channel_decay(chmap, fsurf, basevalue, time_window)
    assert np.all(dryfrac == np.array([[0.75, 0.6875, 0.625, 0.5625, 0.5]]))


def test_planform_olap():
    """Test channel planform overlap."""
    ophi = mob.calculate_planform_overlap(chmap, fsurf, basevalue, time_window)
    assert pytest.approx(ophi == np.array([[1., 0.66666667, 0.33333333,
                                            0., -0.33333333]]))


def test_reworking():
    """Test reworking index."""
    fr = mob.calculate_reworking_fraction(chmap, fsurf, basevalue, time_window)
    assert pytest.approx(fr == np.array([[0., 0.08333333, 0.16666667,
                                          0.25, 0.33333333]]))


def test_channel_abandon():
    """Test channel abandonment function."""
    ch_abandon = mob.calculate_channel_abandonment(chmap, basevalue, time_window)
    assert np.all(ch_abandon == np.array([[0., 0.25, 0.5, 0.75, 1.]]))


def test_channel_presence():
    """Test channel presence."""
    chan_presence = mob.channel_presence(chmap)
    assert np.all(chan_presence == np.array([[0., 0.8, 0.2, 0.],
                                             [0., 0.6, 0.4, 0.],
                                             [0., 0.4, 0.6, 0.],
                                             [0., 0.2, 0.8, 0.]]))
