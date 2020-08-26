import pytest

import sys
import os

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from deltametrics import plot
from deltametrics import cube
from deltametrics import section
from deltametrics import utils

rcm8_path = os.path.join(os.path.dirname(__file__), '..', 'deltametrics',
                         'sample_data', 'files', 'pyDeltaRCM_Output_8.nc')


def test_initialize_default_VariableInfo():
    vi = plot.VariableInfo('testinfo')
    assert vi.cmap.N == 64


def test_initialize_default_VariableInfo_noname():
    with pytest.raises(TypeError):
        vi = plot.VariableInfo()


def test_initialize_default_VariableInfo_name_isstr():
    with pytest.raises(TypeError):
        vi = plot.VariableInfo(None)


def test_initialize_VariableInfo_cmap_str():
    vi = plot.VariableInfo('testinfo', cmap='Blues')
    assert vi.cmap.N == 64
    assert vi.cmap(0)[0] == pytest.approx(0.96862745)


def test_initialize_VariableInfo_cmap_spec():
    vi = plot.VariableInfo('testinfo', cmap=plt.cm.get_cmap('Blues', 7))
    assert vi.cmap.N == 7
    assert vi.cmap(0)[0] == pytest.approx(0.96862745)


def test_initialize_VariableInfo_cmap_tuple():
    vi = plot.VariableInfo('testinfo', cmap=('Blues', 7))
    assert vi.cmap.N == 7
    assert vi.cmap(0)[0] == pytest.approx(0.96862745)


def test_initialize_VariableInfo_label_str():
    vi = plot.VariableInfo('testinfo', label='Test Information')
    assert vi.label == 'Test Information'
    assert vi.name == 'testinfo'


def test_VariableInfo_change_label():
    vi = plot.VariableInfo('testinfo')
    vi.label = 'Test Information'
    assert vi.label == 'Test Information'
    assert vi.name == 'testinfo'


def test_initialize_default_VariableSet():
    vs = plot.VariableSet()
    assert 'eta' in vs.known_list
    assert vs['depth'].vmin == 0


def test_initialize_VariableSet_override_known_VariableInfo():
    vi = plot.VariableInfo('depth')
    od = {'depth': vi}
    vs = plot.VariableSet(override_dict=od)
    assert vs['depth'].vmin is None


def test_initialize_VariableSet_override_unknown_VariableInfo():
    vi = plot.VariableInfo('fakevariable', vmin=-9999)
    od = {'fakevariable': vi}
    vs = plot.VariableSet(override_dict=od)
    assert vs['fakevariable'].vmin == -9999


def test_initialize_VariableSet_override_known_badtype():
    vi = plot.VariableInfo('depth')
    od = ('depth', vi)
    with pytest.raises(TypeError):
        vs = plot.VariableSet(override_dict=od)


def test_VariableSet_add_known_VariableInfo():
    vs = plot.VariableSet()
    vi = plot.VariableInfo('depth', vmin=-9999)
    vs.depth = vi
    assert vs.depth.vmin == -9999


def test_VariableSet_add_unknown_VariableInfo():
    vs = plot.VariableSet()
    vi = plot.VariableInfo('fakevariable', vmin=-9999)
    vs.fakevariable = vi
    assert vs.fakevariable.vmin == -9999


def test_VariableSet_set_known_VariableInfo_direct():
    vs = plot.VariableSet()
    vs.depth.vmin = -9999
    assert vs.depth.vmin == -9999


def test_VariableSet_change_then_default():
    vs = plot.VariableSet()
    _first = vs.depth.cmap(0)[0]
    vi = plot.VariableInfo('depth', vmin=-9999)
    vs.depth = vi
    assert vs.depth.vmin == -9999
    vs.depth = None  # reset to default
    assert vs.depth.cmap(0)[0] == _first
    assert vs.depth.vmin == 0


def test_VariableSet_add_known_badtype():
    vs = plot.VariableSet()
    with pytest.raises(TypeError):
        vs.depth = 'Yellow!'


def test_VariableSet_add_unknown_badtype():
    vs = plot.VariableSet()
    with pytest.raises(TypeError):
        vs.fakevariable = 'Yellow!'


def test_append_colorbar():
    _arr = np.random.randint(0, 100, size=(50, 50))
    fig, ax = plt.subplots()
    im = ax.imshow(_arr)
    cb = plot.append_colorbar(im, ax)
    assert isinstance(cb, matplotlib.colorbar.Colorbar)
    assert ax.use_sticky_edges is False


class TestFillSteps:
    """Test the `_fill_steps` function."""

    arr = np.array([False, False, True, True, False, True,
                    True, True, True, False, True])

    def num_patches(self, pc):
        """hacky util to get length of PatchCollection."""
        return len(pc.properties()['facecolor'])

    def test_return_type(self):
        pc = plot._fill_steps(self.arr)
        assert isinstance(pc, matplotlib.collections.PatchCollection)

    def test_return_length_zero(self):
        _arr = np.array([False])
        pc = plot._fill_steps(_arr)
        assert self.num_patches(pc) == 0

    def test_return_length_zero_trues(self):
        _arr = np.array([True])
        pc = plot._fill_steps(_arr)
        assert self.num_patches(pc) == 0

    def test_return_length_one(self):
        _arr = np.array([False, True])
        pc = plot._fill_steps(_arr)
        assert self.num_patches(pc) == 1

    def test_return_length_three_get_two(self):
        _arr = np.array([False, True, True])
        pc = plot._fill_steps(_arr)
        assert self.num_patches(pc) == 2

    def test_return_length_three_get_two_trues(self):
        _arr = np.array([True, True, True])
        pc = plot._fill_steps(_arr)
        assert self.num_patches(pc) == 2

    def test_return_length_three_get_five(self):
        _arr = np.array([False, True, True, False, False, False,
                         True, True, False, True])
        pc = plot._fill_steps(_arr)
        assert self.num_patches(pc) == 5

    def test_kwargs_default(self):
        pc = plot._fill_steps(self.arr)
        assert self.num_patches(pc) == 7
        _exp = pytest.approx(np.array([0.12156863, 0.46666667, 0.70588235, 1.]))
        assert np.all(pc.get_facecolors()[0] == _exp)

    def test_kwargs_facecolor(self):
        pc = plot._fill_steps(self.arr, facecolor='green')
        assert self.num_patches(pc) == 7
        _exp = pytest.approx(np.array([0., 0.50196078, 0., 1.]))
        assert np.all(pc.get_facecolors()[0] == _exp)


class TestSODTTST:
    """Test the `show_one_dimensional_trajectory_to_strata` function."""

    def test_sodttst_makes_plot(self):
        _e = np.random.randint(0, 10, size=(50,))
        fig, ax = plt.subplots()
        plot.show_one_dimensional_trajectory_to_strata(_e, ax=ax)
        plt.close()

    def test_sodttst_makes_plot_lims_positives(self):
        _e = np.array([0, 1, 4, 5, 4, 10])
        fig, ax = plt.subplots()
        plot.show_one_dimensional_trajectory_to_strata(_e, ax=ax)
        assert ax.get_ylim() == (0, 12)
        plt.close()

    def test_sodttst_makes_plot_lims_negative(self):
        _e = np.array([10, -1, -4, -5, -4, -10])
        fig, ax = plt.subplots()
        plot.show_one_dimensional_trajectory_to_strata(_e, ax=ax)
        assert ax.get_ylim() == (-12, 12)
        plt.close()

    def test_sodttst_makes_plot_lims_negative_zero(self):
        _e = np.array([-1, -1, -4, -5, -4, -10])
        fig, ax = plt.subplots()
        plot.show_one_dimensional_trajectory_to_strata(_e, ax=ax)
        assert ax.get_ylim() == (-12, 0)
        plt.close()

    def test_sodttst_makes_plot_lims_equal(self):
        _e = np.array([-1, -1, -1, -1, -1, -1])
        fig, ax = plt.subplots()
        plot.show_one_dimensional_trajectory_to_strata(_e, ax=ax)
        assert ax.get_ylim() == (-1.2, 0)
        plt.close()

    def test_sodttst_makes_plot_sample_data(self):
        rcm8_path = os.path.join(os.path.dirname(__file__), '..', 'deltametrics',
                         'sample_data', 'files', 'pyDeltaRCM_Output_8.nc')
        rcm8cube = cube.DataCube(rcm8_path)
        locs = np.array([[48, 152], [8, 63], [14, 102], [92, 218], [102, 168],
                         [26, 114], [62, 135], [61, 201], [65, 193], [23, 175]])
        for i in range(10):
            _e = rcm8cube['eta'][:, locs[i, 0], locs[i, 1]]
            fig, ax = plt.subplots()
            plot.show_one_dimensional_trajectory_to_strata(_e, ax=ax)
            plt.close()

    def test_sodttst_makes_plot_no_ax(self):
        _e = np.random.randint(0, 10, size=(50,))
        plot.show_one_dimensional_trajectory_to_strata(_e)
        plt.close()

    def test_sodttst_makes_plot_3d_column(self):
        _e = np.random.randint(0, 10, size=(50,1,1))
        plot.show_one_dimensional_trajectory_to_strata(_e)
        plt.close()

    def test_sodttst_makes_plot_2d_column_error(self):
        _e = np.random.randint(0, 10, size=(50,100,1))
        with pytest.raises(ValueError, match=r'Elevation data "e" must *.'):
            plot.show_one_dimensional_trajectory_to_strata(_e)
        plt.close()


class TestGetDisplayArrays:

    rcm8cube_nostrat = cube.DataCube(rcm8_path)
    rcm8cube_nostrat.register_section('test', section.StrikeSection(y=5))
    dsv_nostrat = rcm8cube_nostrat.sections['test']['velocity']

    rcm8cube = cube.DataCube(rcm8_path)
    rcm8cube.stratigraphy_from('eta')
    rcm8cube.register_section('test', section.StrikeSection(y=5))
    dsv = rcm8cube.sections['test']['velocity']

    sc8cube = cube.StratigraphyCube.from_DataCube(rcm8cube)
    sc8cube.register_section('test', section.StrikeSection(y=5))
    ssv = sc8cube.sections['test']['velocity']

    def test_dsv_nostrat_get_display_arrays_spacetime(self):
        _data, _X, _Y = plot.get_display_arrays(self.dsv_nostrat,
                                                data='spacetime')
        assert (_data.shape == _X.shape) and (_data.shape == _Y.shape)
        assert np.all(_data == self.dsv_nostrat)

    def test_dsv_nostrat_get_display_arrays_preserved(self):
        with pytest.raises(utils.NoStratigraphyError):
            plot.get_display_arrays(self.dsv_nostrat,
                                    data='preserved')

    def test_dsv_nostrat_get_display_arrays_stratigraphy(self):
        with pytest.raises(utils.NoStratigraphyError):
            plot.get_display_arrays(self.dsv_nostrat,
                                    data='stratigraphy')

    def test_dsv_get_display_arrays_spacetime(self):
        _data, _X, _Y = plot.get_display_arrays(self.dsv,
                                                data='spacetime')
        assert (_data.shape == _X.shape) and (_data.shape == _Y.shape)
        assert np.all(_data == self.dsv)

    def test_dsv_get_display_arrays_preserved(self):
        _data, _X, _Y = plot.get_display_arrays(self.dsv,
                                                data='preserved')
        assert (_data.shape == _X.shape) and (_data.shape == _Y.shape)
        assert np.any(~_data._mask)  # check that some are False

    def test_dsv_get_display_arrays_stratigraphy(self):
        _data, _X, _Y = plot.get_display_arrays(self.dsv,
                                                data='stratigraphy')
        assert (_data.shape[0] == _X.shape[0] - 1)
        assert (_data.shape[1] == _Y.shape[1] - 1)
        assert (_data.shape[0] == _X.shape[0] - 1)
        assert (_data.shape[1] == _Y.shape[1] - 1)

    def test_ssv_get_display_arrays_spacetime(self):
        with pytest.raises(AttributeError, match=r'No "spacetime" or "preserved"*.'):
            _data, _X, _Y = plot.get_display_arrays(self.ssv,
                                                    data='spacetime')

    def test_ssv_get_display_arrays_preserved(self):
        with pytest.raises(AttributeError, match=r'No "spacetime" or "preserved"*.'):
            _data, _X, _Y = plot.get_display_arrays(self.ssv,
                                                    data='preserved')

    def test_ssv_get_display_arrays_stratigraphy(self):
        _data, _X, _Y = plot.get_display_arrays(self.ssv,
                                                data='stratigraphy')
        assert (_data.shape == _X.shape) and (_data.shape == _Y.shape)


class TestGetDisplayLines:

    rcm8cube_nostrat = cube.DataCube(rcm8_path)
    rcm8cube_nostrat.register_section('test', section.StrikeSection(y=5))
    dsv_nostrat = rcm8cube_nostrat.sections['test']['velocity']

    rcm8cube = cube.DataCube(rcm8_path)
    rcm8cube.stratigraphy_from('eta')
    rcm8cube.register_section('test', section.StrikeSection(y=5))
    dsv = rcm8cube.sections['test']['velocity']

    sc8cube = cube.StratigraphyCube.from_DataCube(rcm8cube)
    sc8cube.register_section('test', section.StrikeSection(y=5))
    ssv = sc8cube.sections['test']['velocity']

    def test_dsv_nostrat_get_display_lines_spacetime(self):
        _data, _segments = plot.get_display_lines(self.dsv_nostrat,
                                                  data='spacetime')
        assert _segments.shape[1:] == (2, 2)

    def test_dsv_nostrat_get_display_lines_preserved(self):
        with pytest.raises(utils.NoStratigraphyError):
            plot.get_display_lines(self.dsv_nostrat,
                                   data='preserved')

    def test_dsv_nostrat_get_display_lines_stratigraphy(self):
        with pytest.raises(utils.NoStratigraphyError):
            plot.get_display_lines(self.dsv_nostrat,
                                   data='stratigraphy')

    def test_dsv_get_display_lines_spacetime(self):
        _data, _segments = plot.get_display_lines(self.dsv,
                                                  data='spacetime')
        assert _segments.shape[1:] == (2, 2)

    def test_dsv_get_display_lines_preserved(self):
        _data, _segments = plot.get_display_lines(self.dsv,
                                                  data='preserved')
        assert _segments.shape[1:] == (2, 2)

    def test_dsv_get_display_lines_stratigraphy(self):
        _data, _segments = plot.get_display_lines(self.dsv,
                                                  data='stratigraphy')
        assert _segments.shape[1:] == (2, 2)

    def test_ssv_get_display_lines_spacetime(self):
        with pytest.raises(AttributeError, match=r'No "spacetime" or "preserved"*.'):
            _data, _segments = plot.get_display_lines(self.ssv,
                                                      data='spacetime')

    def test_ssv_get_display_lines_preserved(self):
        with pytest.raises(AttributeError, match=r'No "spacetime" or "preserved"*.'):
            plot.get_display_lines(self.ssv,
                                   data='preserved')

    @pytest.mark.xfail(raises=NotImplementedError, strict=True,
                       reason='Have not determined how to implement yet.')
    def test_ssv_get_display_lines_stratigraphy(self):
        plot.get_display_lines(self.ssv,
                               data='stratigraphy')


class TestGetDisplayLimits:

    rcm8cube_nostrat = cube.DataCube(rcm8_path)
    rcm8cube_nostrat.register_section('test', section.StrikeSection(y=5))
    dsv_nostrat = rcm8cube_nostrat.sections['test']['velocity']

    rcm8cube = cube.DataCube(rcm8_path)
    rcm8cube.stratigraphy_from('eta')
    rcm8cube.register_section('test', section.StrikeSection(y=5))
    dsv = rcm8cube.sections['test']['velocity']

    sc8cube = cube.StratigraphyCube.from_DataCube(rcm8cube)
    sc8cube.register_section('test', section.StrikeSection(y=5))
    ssv = sc8cube.sections['test']['velocity']

    def test_dsv_nostrat_get_display_limits_spacetime(self):
        _lims = plot.get_display_limits(self.dsv_nostrat, data='spacetime')
        assert len(_lims) == 4

    def test_dsv_nostrat_get_display_limits_preserved(self):
        with pytest.raises(utils.NoStratigraphyError):
            plot.get_display_limits(self.dsv_nostrat, data='preserved')

    def test_dsv_nostrat_get_display_limits_stratigraphy(self):
        with pytest.raises(utils.NoStratigraphyError):
            plot.get_display_limits(self.dsv_nostrat, data='stratigraphy')

    def test_dsv_get_display_limits_spacetime(self):
        _lims = plot.get_display_limits(self.dsv, data='spacetime')
        assert len(_lims) == 4

    def test_dsv_get_display_limits_preserved(self):
        _lims = plot.get_display_limits(self.dsv, data='preserved')
        assert len(_lims) == 4

    def test_dsv_get_display_limits_stratigraphy(self):
        _lims = plot.get_display_limits(self.dsv, data='stratigraphy')
        assert len(_lims) == 4

    def test_ssv_get_display_limits_spacetime(self):
        with pytest.raises(AttributeError, match=r'No "spacetime" or "preserved"*.'):
            _lims = plot.get_display_limits(self.ssv, data='spacetime')

    def test_ssv_get_display_limits_preserved(self):
        with pytest.raises(AttributeError, match=r'No "spacetime" or "preserved"*.'):
            _lims = plot.get_display_limits(self.ssv, data='preserved')

    def test_ssv_get_display_limits_stratigraphy(self):
        _lims = plot.get_display_limits(self.ssv, data='stratigraphy')
        assert len(_lims) == 4
