"""Tests for mobility.py."""
import pytest
import sys
import os
import numpy as np
import xarray as xr

import deltametrics as dm
from deltametrics import cube
from deltametrics import mobility as mob
from deltametrics.sample_data import _get_rcm8_path


rcm8_path = _get_rcm8_path()
with pytest.warns(UserWarning):
    rcm8cube = cube.DataCube(rcm8_path)

# define some masks once up top
chmask = []
landmask = []
for i in range(20, 23):
    chmask.append(
        dm.mask.ChannelMask(rcm8cube['eta'][i, :, :],
                            rcm8cube['velocity'][i, :, :],
                            elevation_threshold=0,
                            flow_threshold=0.3))
    landmask.append(
        dm.mask.LandMask(rcm8cube['eta'][i, :, :],
                         elevation_threshold=0))

# make them into xarrays (list of xarrays)
dims = ('time', 'x', 'y')  # assumes an ultimate t-x-y shape
coords = {'time': np.arange(1),
          'x': np.arange(chmask[0].mask.shape[0]),
          'y': np.arange(chmask[0].mask.shape[1])}
# channel masks
ch_xarr = [xr.DataArray(
    data=np.reshape(chmask[i].mask.data,
                    (1, chmask[i].mask.shape[0], chmask[i].mask.shape[1])),
    coords=coords, dims=dims)
            for i in range(len(chmask))]
# land masks
land_xarr = [xr.DataArray(
    data=np.reshape(landmask[i].mask.data,
                    (1, landmask[i].mask.shape[0], landmask[i].mask.shape[1])),
    coords=coords, dims=dims)
            for i in range(len(landmask))]

# convert them to ndarrays
ch_arr = np.zeros((3, chmask[0].shape[0], chmask[0].shape[1]))
land_arr = np.zeros((3, chmask[0].shape[0], chmask[0].shape[1]))
ch_arr_list = []
land_arr_list = []
for i in range(3):
    ch_arr[i, ...] = ch_xarr[i].data
    land_arr[i, ...] = land_xarr[i].data
    ch_arr_list.append(ch_xarr[i].data.squeeze())
    land_arr_list.append(land_xarr[i].data.squeeze())


def test_check_input_list_of_mask():
    """Test that a deltametrics.mask.BaseMask type can be used."""
    # call checker function
    assert isinstance(chmask, list)
    chmap, landmap, basevalues, time_window, dim0 = mob.check_inputs(
        chmask, basevalues_idx=[0], window_idx=1, landmap=landmask)
    # assert types
    assert dim0 == 'time'
    assert isinstance(chmap, xr.DataArray) is True
    assert isinstance(landmap, xr.DataArray) is True
    assert len(np.shape(chmap)) == 3
    assert len(np.shape(landmap)) == 3
    assert isinstance(basevalues, list) is True
    assert isinstance(time_window, int) is True


def test_check_input_single_mask_error():
    """Test that a deltametrics.mask.BaseMask type can be used."""
    # call checker function
    with pytest.raises(TypeError, match=r'Cannot input a Mask .*'):
        chmap, landmap, basevalues, time_window, dim0 = mob.check_inputs(
            chmask[0], basevalues_idx=[0], window_idx=1,
            landmap=landmask[0])


def test_check_xarrays():
    """Test that an xarray.DataArray can be used as an input."""
    # call checker function
    chmap, landmap, basevalues, time_window, dim0 = mob.check_inputs(
        ch_xarr, basevalues_idx=[0], window_idx=1,
        landmap=land_xarr)
    # assert types
    assert dim0 == 'time'
    assert isinstance(chmap, xr.DataArray) is True
    assert isinstance(landmap, xr.DataArray) is True
    assert isinstance(basevalues, list) is True
    assert isinstance(time_window, int) is True


def test_check_list_ndarrays():
    """Test that a list of numpy.ndarray can be used as an input."""
    # call checker function
    chmap, landmap, basevalues, time_window, dim0 = mob.check_inputs(
        ch_arr_list, basevalues_idx=[0], window_idx=1,
        landmap=land_arr_list)
    # assert types
    assert dim0 == 'time'
    assert isinstance(chmap, xr.DataArray) is True
    assert isinstance(landmap, xr.DataArray) is True
    assert isinstance(basevalues, list) is True
    assert isinstance(time_window, int) is True


def test_check_ndarrays():
    """Test that a numpy.ndarray can be used as an input."""
    # call checker function
    chmap, landmap, basevalues, time_window, dim0 = mob.check_inputs(
        ch_arr, basevalues_idx=[0], window_idx=1,
        landmap=land_arr)
    # assert types
    assert dim0 == 'time'
    assert isinstance(chmap, xr.DataArray) is True
    assert isinstance(landmap, xr.DataArray) is True
    assert isinstance(basevalues, list) is True
    assert isinstance(time_window, int) is True


def test_check_basevalues_window():
    """Test that basevalues and window inputs work."""
    # call checker function
    chmap, landmap, basevalues, time_window, dim0 = mob.check_inputs(
        ch_arr, basevalues=[0], window=1, landmap=land_arr)
    # assert types
    assert dim0 == 'time'
    assert isinstance(chmap, xr.DataArray) is True
    assert isinstance(landmap, xr.DataArray) is True
    assert isinstance(basevalues, list) is True
    assert isinstance(time_window, int) is True


def test_check_input_nolandmask():
    """Test that the check input can run without a landmap."""
    # call checker function
    chmap, landmap, basevalues, time_window, dim0 = mob.check_inputs(
        chmask, basevalues_idx=[0], window_idx=1)
    # assert types
    assert dim0 == 'time'
    assert isinstance(chmap, xr.DataArray) is True
    assert landmap is None
    assert isinstance(basevalues, list) is True
    assert isinstance(time_window, int) is True


@pytest.mark.xfail(reason='Removed binary check - to be added back later.')
def test_check_input_notbinary_chmap():
    """Test that nonbinary channel map raises error."""
    ch_nonbin = np.zeros((3, 3, 3))
    ch_nonbin[0, 1, 1] = 1
    ch_nonbin[0, 1, 2] = 2
    with pytest.raises(ValueError):
        mob.check_inputs(ch_nonbin, basevalues_idx=[0], window_idx=1)


@pytest.mark.xfail(reason='Removed binary check - to be added back later.')
def test_check_input_notbinary_landmap():
    """Test that nonbinary land map raises error."""
    land_nonbin = np.zeros((3, 3, 3))
    land_nonbin[0, 1, 1] = 1
    land_nonbin[0, 1, 2] = 2
    ch_bin = np.zeros_like(land_nonbin)
    with pytest.raises(ValueError):
        mob.check_inputs(ch_bin, basevalues_idx=[0], window_idx=1,
                         landmap=land_nonbin)


def test_check_input_invalid_chmap():
    """Test that an invalid channel map input will throw an error."""
    with pytest.raises(TypeError):
        mob.check_inputs('invalid', basevalues_idx=[0], window_idx=1,
                         landmap=landmask)


def test_check_input_invalid_landmap():
    """Test that an invalid landmap will throw an error."""
    with pytest.raises(TypeError):
        mob.check_inputs(chmask, basevalues_idx=[0], window_idx=1,
                         landmap='invalid')


def test_check_input_invalid_basevalues():
    """Test that a non-listable basevalues throws an error."""
    with pytest.raises(TypeError):
        mob.check_inputs(chmask, basevalues=0, window_idx='invalid')


def test_check_input_invalid_basevalues_idx():
    """Test that a non-listable basevalues_idx throws an error."""
    with pytest.raises(TypeError):
        mob.check_inputs(chmask, basevalues_idx=0, window_idx='invalid')


def test_check_no_basevalues_error():
    """No basevalues will throw an error."""
    with pytest.raises(ValueError):
        mob.check_inputs(chmask, window_idx='invalid')


def test_check_input_invalid_time_window():
    """Test that a non-valid time_window throws an error."""
    with pytest.raises(TypeError):
        mob.check_inputs(chmask, basevalues_idx=[0], window='invalid')


def test_check_input_invalid_time_window_idx():
    """Test that a non-valid time_window_idx throws an error."""
    with pytest.raises(TypeError):
        mob.check_inputs(chmask, basevalues_idx=[0], window_idx='invalid')


def test_check_no_time_window():
    """Test that no time_window throws an error."""
    with pytest.raises(ValueError):
        mob.check_inputs(chmask, basevalues_idx=[0])


def test_check_input_2dchanmask():
    """Test that an unexpected channel mask shape throws an error."""
    with pytest.raises(TypeError):
        mob.check_inputs(np.ones((5,)), basevalues_idx=[0], window_idx=1)


def test_check_input_diff_shapes():
    """Test that differently shaped channel and land masks throw an error."""
    with pytest.raises(ValueError):
        mob.check_inputs(chmask, basevalues_idx=[0], window_idx=1,
                         landmap=np.ones((3, 3, 3)))


def test_check_input_1dlandmask():
    """Test a 1d landmask that will throw an error."""
    with pytest.raises(TypeError):
        mob.check_inputs(chmask, basevalues_idx=[0], window_idx=1,
                         landmap=np.ones((10, 1)))


def test_check_input_exceedmaxvals():
    """Test a basevalue + time window combo that exceeds time indices."""
    with pytest.raises(ValueError):
        mob.check_inputs(chmask, basevalues_idx=[0], window_idx=100)


def test_check_input_invalid_list():
    """Test a wrong list."""
    with pytest.raises(TypeError):
        mob.check_inputs(['str', 5, 1.], basevalues_idx=[0], window_idx=1)


def test_check_input_list_wrong_shape():
    """Test list with wrongly shaped arrays."""
    in_list = [np.zeros((5, 2, 1, 1)), np.zeros((5, 2, 2, 2))]
    with pytest.raises(ValueError):
        mob.check_inputs(in_list, basevalues_idx=[0], window_idx=100)


@pytest.mark.xfail(
    reason='Removed this functionality - do we want to blindly expand dims?')
def test_check_input_castlandmap():
    """Test ability to cast a 2D landmask to match 3D channelmap."""
    chmap, landmap, bv, tw, dim0 = mob.check_inputs(
        chmask, basevalues_idx=[0], window_idx=1,
        landmap=landmask[0].mask[:, :])
    assert np.shape(chmap) == np.shape(landmap)


# Test actual mobility functions using a simple example
# domain is a small 4x4 region with a single 4 cell channel
# every timestep, a cell from the channel moves over a column
# colab notebook:
#   https://colab.research.google.com/drive/1-lkswSZSGRkhsArLm245WMSiKvj03YyU?usp=sharing

# first define the domain and channel maps etc.
chmap = np.zeros((5, 4, 4))
# define time = 0
chmap[0, :, 1] = 1
# every time step one cell of the channel will migrate one pixel to the right
for i in range(1, 5):
    chmap[i, :, :] = chmap[i-1, :, :].copy()
    chmap[i, -1*i, 1] = 0
    chmap[i, -1*i, 2] = 1
# alt typing for the input map
# chmap xarrays
dims = ('time', 'x', 'y')  # assumes an ultimate t-x-y shape
coords = {'time': np.arange(5),
          'x': np.arange(chmap.shape[1]),
          'y': np.arange(chmap.shape[2])}
chmap_xr = xr.DataArray(data=chmap, coords=coords, dims=dims)
# lists
chmap_xr_list = []
chmap_list = []
for i in range(5):
    chmap_list.append(chmap[i, ...])
    chmap_xr_list.append(chmap_xr[i, ...].squeeze())


# define the fluvial surface - entire 4x4 area through all time
fsurf = np.ones((5, 4, 4))
# define the index corresponding to the basemap at time 0
basevalue = [0]
# define the size of the time window to use
time_window = 5


def test_dry_decay():
    """Test dry fraction decay."""
    dryfrac = mob.calculate_channel_decay(
        chmap, fsurf, basevalues_idx=basevalue, window_idx=time_window)
    assert np.all(dryfrac == np.array([[0.75, 0.6875, 0.625, 0.5625, 0.5]]))


def test_planform_olap():
    """Test channel planform overlap."""
    ophi = mob.calculate_planform_overlap(
        chmap, fsurf, basevalues_idx=basevalue, window_idx=time_window)
    assert pytest.approx(ophi.values) == np.array([[1., 0.66666667, 0.33333333,
                                                    0., -0.33333333]])


def test_reworking():
    """Test reworking index."""
    fr = mob.calculate_reworking_fraction(
        chmap, fsurf, basevalues_idx=basevalue, window_idx=time_window)
    assert pytest.approx(fr.values) == np.array([[0., 0.08333333, 0.16666667,
                                                  0.25, 0.33333333]])


def test_channel_abandon():
    """Test channel abandonment function."""
    ch_abandon = mob.calculate_channel_abandonment(
        chmap, basevalues_idx=basevalue, window_idx=time_window)
    assert np.all(ch_abandon == np.array([[0., 0.25, 0.5, 0.75, 1.]]))


def test_channel_presence():
    """Test channel presence with a regular array."""
    chan_presence = mob.channel_presence(chmap)
    assert np.all(chan_presence == np.array([[0., 0.8, 0.2, 0.],
                                             [0., 0.6, 0.4, 0.],
                                             [0., 0.4, 0.6, 0.],
                                             [0., 0.2, 0.8, 0.]]))


def test_channel_presence_xarray():
    """Test channel presence with an xarray."""
    chan_presence = mob.channel_presence(chmap_xr)
    assert np.all(chan_presence == np.array([[0., 0.8, 0.2, 0.],
                                             [0., 0.6, 0.4, 0.],
                                             [0., 0.4, 0.6, 0.],
                                             [0., 0.2, 0.8, 0.]]))


def test_channel_presence_xarray_list():
    """Test channel presence with a list of xarrays."""
    chan_presence = mob.channel_presence(chmap_xr_list)
    assert np.all(chan_presence == np.array([[0., 0.8, 0.2, 0.],
                                             [0., 0.6, 0.4, 0.],
                                             [0., 0.4, 0.6, 0.],
                                             [0., 0.2, 0.8, 0.]]))


def test_channel_presence_array_list():
    """Test channel presence with a list of arrays."""
    chan_presence = mob.channel_presence(chmap_list)
    assert np.all(chan_presence == np.array([[0., 0.8, 0.2, 0.],
                                             [0., 0.6, 0.4, 0.],
                                             [0., 0.4, 0.6, 0.],
                                             [0., 0.2, 0.8, 0.]]))


def test_invalid_list_channel_presence():
    """Test an invalid list."""
    with pytest.raises(ValueError):
        mob.channel_presence(['in', 'valid', 'list'])


def test_invalid_type_channel_presence():
    """Test an invalid input typing."""
    with pytest.raises(TypeError):
        mob.channel_presence('invalid type input')


def test_calculate_crv_mag_zeros():
    """Test calculate crv magnitude with zeros."""
    z_arr = np.zeros((5, 3, 2))
    z_mag = mob.calculate_crv_mag(z_arr, normalize_output=False)
    assert np.all(z_mag == 0.0)


def test_calculate_crv_mag_random():
    """Test calculate crv magnitude with random."""
    z_arr = np.random.rand(25, 3, 2)
    z_mag = mob.calculate_crv_mag(z_arr, normalize_output=True)
    assert np.all(z_mag > 0.0)
    assert np.all(z_mag <= 1.0)


def test_calculate_noslopes():
    """Test calculate slopes w/ constants."""
    arr = np.zeros((5, 3, 2))
    slopes = mob.calculate_slopes(arr)
    assert np.all(slopes == 0.0)


def test_calculate_neg_slopes():
    """Test calculate slopes w/ negative values."""
    arr = np.linspace(np.ones((5, 3)), np.zeros((5, 3)), 5)
    slopes = mob.calculate_slopes(arr)
    assert np.all(slopes < 0.0)


def test_calculate_pos_slopes():
    """Test calculate slopes w/ positive values."""
    arr = np.linspace(np.zeros((3, 2)), np.ones((3, 2)), 5)
    slopes = mob.calculate_slopes(arr)
    assert np.all(slopes > 0.0)


def test_calculate_dir_crv():
    """Test calculate directional crv."""
    # crv magnitude array
    crv_mag = np.ones((3, 2)) * 10.0
    crv_mag[:, 0] = 1.0
    # slope array
    slopes = np.ones((3, 2))
    slopes[0, :] = -1.0
    # function
    dir_crv = mob.calculate_directional_crv(crv_mag, slopes, 0.0)
    assert np.all(dir_crv[0, :] < 0.0)
    assert np.all(dir_crv[1:, :] > 0.0)
    assert np.all(np.abs(dir_crv)[:, 0] == 1.0)
    assert np.all(np.abs(dir_crv)[:, 1] == 10.0)


def test_calculate_crv():
    """Testing the overall crv calculation function."""
    # data array
    arr = np.ones((2, 3, 2)) * 10.0
    arr[0, :, :] = 1.0
    arr[1, 0, :] = -5.0
    # function
    crv_mag, slopes, dir_crv = mob.calculate_crv(arr, threshold=0.0)
    assert np.all(crv_mag == np.abs(dir_crv))
    assert np.all(slopes[0, :] < 0.0)
    assert np.all(slopes[1:, :] > 0.0)
    assert np.all(dir_crv[0, :] < 0.0)
    assert np.all(dir_crv[1:, :] > 0.0)