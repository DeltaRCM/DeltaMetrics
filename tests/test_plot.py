import pytest

import os

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from deltametrics import plot
from deltametrics import cube
from deltametrics import section
from deltametrics import utils
from deltametrics.sample_data import _get_golf_path


golf_path = _get_golf_path()


class TestVariableInfo:

    def test_initialize_default_VariableInfo(self):
        vi = plot.VariableInfo('testinfo')
        assert vi.cmap.N == 64

    def test_initialize_default_VariableInfo_noname(self):
        with pytest.raises(TypeError):
            _ = plot.VariableInfo()

    def test_initialize_default_VariableInfo_name_isstr(self):
        with pytest.raises(TypeError):
            _ = plot.VariableInfo(None)

    def test_initialize_VariableInfo_cmap_str(self):
        vi = plot.VariableInfo('testinfo', cmap='Blues')
        assert vi.cmap.N == 64
        assert vi.cmap(0)[0] == pytest.approx(0.96862745)

    def test_initialize_VariableInfo_cmap_spec(self):
        vi = plot.VariableInfo('testinfo', cmap=plt.cm.get_cmap('Blues', 7))
        assert vi.cmap.N == 7
        assert vi.cmap(0)[0] == pytest.approx(0.96862745)

    def test_initialize_VariableInfo_cmap_tuple(self):
        vi = plot.VariableInfo('testinfo', cmap=('Blues', 7))
        assert vi.cmap.N == 7
        assert vi.cmap(0)[0] == pytest.approx(0.96862745)

    def test_initialize_VariableInfo_label_str(self):
        vi = plot.VariableInfo('testinfo', label='Test Information')
        assert vi.label == 'Test Information'
        assert vi.name == 'testinfo'

    def test_VariableInfo_change_label(self):
        vi = plot.VariableInfo('testinfo')
        vi.label = 'Test Information'
        assert vi.label == 'Test Information'
        assert vi.name == 'testinfo'
        with pytest.raises(TypeError):
            vi.label = True
        with pytest.raises(TypeError):
            vi.label = 19

    def test_VariableInfo_change_cmap(self):
        vi = plot.VariableInfo('testinfo')
        _jet = matplotlib.cm.get_cmap('jet', 21)
        vi.cmap = _jet
        assert vi.cmap == _jet
        assert vi.cmap.N == 21
        vi.cmap = ('Blues', 7)
        assert vi.cmap.N == 7
        assert vi.cmap(0)[0] == pytest.approx(0.96862745)
        with pytest.raises(TypeError):
            vi.cmap = True
        with pytest.raises(TypeError):
            vi.label = 19


class TestVariableSet:

    def test_initialize_default_VariableSet(self):
        vs = plot.VariableSet()
        assert 'eta' in vs.known_list
        assert vs['depth'].vmin == 0
        assert vs.variables == vs.known_list
        # get returns from some properties anc check types
        assert isinstance(vs.variables, list)
        assert isinstance(vs.x, plot.VariableInfo)
        assert isinstance(vs.y, plot.VariableInfo)
        assert isinstance(vs.eta, plot.VariableInfo)
        assert isinstance(vs.stage, plot.VariableInfo)
        assert isinstance(vs.depth, plot.VariableInfo)
        assert isinstance(vs.discharge, plot.VariableInfo)
        assert isinstance(vs.velocity, plot.VariableInfo)
        assert isinstance(vs.strata_sand_frac, plot.VariableInfo)
        assert isinstance(vs.sedflux, plot.VariableInfo)

    def test_initialize_VariableSet_override_known_VariableInfo(self):
        vi = plot.VariableInfo('depth')
        od = {'depth': vi}
        vs = plot.VariableSet(override_dict=od)
        assert vs['depth'].vmin is None

    def test_initialize_VariableSet_override_unknown_VariableInfo(self):
        vi = plot.VariableInfo('fakevariable', vmin=-9999)
        od = {'fakevariable': vi}
        vs = plot.VariableSet(override_dict=od)
        assert vs['fakevariable'].vmin == -9999

    def test_initialize_VariableSet_override_known_badtype(self):
        vi = plot.VariableInfo('depth')
        od = ('depth', vi)
        with pytest.raises(TypeError):
            _ = plot.VariableSet(override_dict=od)

    def test_VariableSet_add_known_VariableInfo(self):
        vs = plot.VariableSet()
        vi = plot.VariableInfo('depth', vmin=-9999)
        vs.depth = vi
        assert vs.depth.vmin == -9999
        assert vs['depth'].vmin == -9999

    def test_VariableSet_add_unknown_VariableInfo(self):
        vs = plot.VariableSet()
        vi = plot.VariableInfo('fakevariable', vmin=-9999)
        vs.fakevariable = vi
        assert vs.fakevariable.vmin == -9999
        assert 'fakevariable' in vs.variables
        assert vs['fakevariable'].vmin == -9999

    def test_VariableSet_set_known_VariableInfo_direct(self):
        vs = plot.VariableSet()
        vs.depth.vmin = -9999
        assert vs.depth.vmin == -9999
        assert vs['depth'].vmin == -9999

    def test_VariableSet_change_then_default(self):
        vs = plot.VariableSet()
        _first = vs.depth.cmap(0)[0]
        vi = plot.VariableInfo('depth', vmin=-9999)
        vs.depth = vi
        assert vs.depth.vmin == -9999
        vs.depth = None  # reset to default
        assert vs.depth.cmap(0)[0] == _first
        assert vs.depth.vmin == 0

    def test_VariableSet_add_known_badtype(self):
        vs = plot.VariableSet()
        with pytest.raises(TypeError):
            vs.depth = 'Yellow!'

    def test_VariableSet_add_unknown_badtype(self):
        vs = plot.VariableSet()
        with pytest.raises(TypeError):
            vs.fakevariable = 'Yellow!'

    def test_get_unknown_notadded_variable(self):
        # should return a default VariableInfo with name field
        vs = plot.VariableSet()
        got = vs['fakevariable']
        assert got.name == 'fakevariable'
        # NOTE: this does not work with attribute accessing
        with pytest.raises(AttributeError):
            _ = vs.fakevariable


class TestAppendColorbar:

    def test_append_colorbar_working(self):
        """Test that the routine works.
        Doesn't really make any meaningful assertions.
        """
        _arr = np.random.randint(0, 100, size=(50, 50))
        fig, ax = plt.subplots()
        im = ax.imshow(_arr)
        cb = plot.append_colorbar(im, ax)
        assert isinstance(cb, matplotlib.colorbar.Colorbar)
        assert ax.use_sticky_edges is False

    def test_size_argument_passed(self):
        """Test that the routine works.
        Doesn't really make any meaningful assertions.
        """
        _arr = np.random.randint(0, 100, size=(50, 50))
        fig, ax = plt.subplots()
        im = ax.imshow(_arr)
        cb = plot.append_colorbar(im, ax, size=10)
        assert isinstance(cb, matplotlib.colorbar.Colorbar)

    def test_kwargs_argument_passed(self):
        _arr = np.random.randint(0, 100, size=(50, 50))
        fig, ax = plt.subplots()
        im = ax.imshow(_arr)
        _formatter = plt.FuncFormatter(lambda val, loc: np.round(val, 0))
        cb = plot.append_colorbar(im, ax, size=10, format=_formatter)
        assert isinstance(cb, matplotlib.colorbar.Colorbar)
        assert cb.formatter is _formatter


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
        _exp = pytest.approx(np.array([0.12156863, 0.46666667,
                                       0.70588235, 1.]))
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
        plot.show_one_dimensional_trajectory_to_strata(
            _e, ax=ax, dz=0.1)
        plt.close()

    def test_sodttst_makes_labeled_strata(self):
        _e = np.random.randint(0, 10, size=(50,))
        fig, ax = plt.subplots()
        plot.show_one_dimensional_trajectory_to_strata(
            _e, ax=ax, dz=0.1, label_strata=False)
        plt.close()

    def test_sodttst_makes_plot_lims_positives(self):
        _e = np.array([0, 1, 4, 5, 4, 10])
        fig, ax = plt.subplots()
        plot.show_one_dimensional_trajectory_to_strata(
            _e, ax=ax, dz=0.1)
        assert ax.get_ylim() == (0, 12)
        plt.close()

    def test_sodttst_makes_plot_lims_negative(self):
        _e = np.array([10, -1, -4, -5, -4, -10])
        fig, ax = plt.subplots()
        plot.show_one_dimensional_trajectory_to_strata(
            _e, ax=ax, dz=0.1)
        assert ax.get_ylim() == (-12, 12)
        plt.close()

    def test_sodttst_makes_plot_lims_negative_zero(self):
        _e = np.array([-1, -1, -4, -5, -4, -10])
        fig, ax = plt.subplots()
        plot.show_one_dimensional_trajectory_to_strata(
            _e, ax=ax, dz=0.1)
        assert ax.get_ylim() == (-12, 0)
        plt.close()

    def test_sodttst_makes_plot_lims_equal(self):
        _e = np.array([-1, -1, -1, -1, -1, -1])
        fig, ax = plt.subplots()
        plot.show_one_dimensional_trajectory_to_strata(
            _e, ax=ax, dz=0.1)
        assert ax.get_ylim() == (-1.2, 0)
        plt.close()

    def test_sodttst_makes_plot_sample_data(self):
        rcm8cube = cube.DataCube(golf_path)
        locs = np.array([[48, 152], [8, 63], [14, 102],
                         [92, 118], [92, 168], [26, 114],
                         [62, 135], [61, 171], [65, 193],
                         [23, 175]])
        for i in range(10):
            _e = rcm8cube['eta'][:, locs[i, 0], locs[i, 1]]
            fig, ax = plt.subplots()
            plot.show_one_dimensional_trajectory_to_strata(
                _e, ax=ax, dz=0.1)
            plt.close()

    def test_sodttst_makes_plot_no_ax(self):
        _e = np.random.randint(0, 10, size=(50,))
        plot.show_one_dimensional_trajectory_to_strata(_e, dz=0.1)
        plt.close()

    def test_sodttst_makes_plot_3d_column(self):
        _e = np.random.randint(0, 10, size=(50, 1, 1))
        plot.show_one_dimensional_trajectory_to_strata(_e, dz=0.1)
        plt.close()

    def test_sodttst_makes_plot_2d_column_error(self):
        _e = np.random.randint(0, 10, size=(50, 100, 1))
        with pytest.raises(ValueError, match=r'Elevation data "e" must *.'):
            plot.show_one_dimensional_trajectory_to_strata(_e, dz=0.1)
        plt.close()

    def test_sodttst_makes_plot_subsidence_zeros(self):
        _e = np.array([0, 1, 4, 5, 4, 10])
        _s = np.array([0, 0, 0, 0, 0, 0])
        fig, ax = plt.subplots()
        plot.show_one_dimensional_trajectory_to_strata(
            _e, sigma_dist=_s, ax=ax, dz=0.1)
        assert ax.get_ylim() == (0, 12)
        plt.close()

    def test_sodttst_makes_plot_lims_equal_via_subs(self):
        _e = np.arange(0, 6)
        fig, ax = plt.subplots()
        plot.show_one_dimensional_trajectory_to_strata(
            _e, sigma_dist=1.0, ax=ax, dz=0.1)
        assert ax.get_ylim()[0] < 0
        assert ax.get_ylim()[1] > 0
        plt.close()


class TestGetDisplayArrays:

    rcm8cube_nostrat = cube.DataCube(golf_path)
    rcm8cube_nostrat.register_section(
        'test', section.StrikeSection(distance_idx=5))
    dsv_nostrat = rcm8cube_nostrat.sections['test']['velocity']

    rcm8cube = cube.DataCube(golf_path)
    rcm8cube.stratigraphy_from('eta', dz=0.1)
    rcm8cube.register_section(
        'test', section.StrikeSection(distance_idx=5))
    dsv = rcm8cube.sections['test']['velocity']

    sc8cube = cube.StratigraphyCube.from_DataCube(rcm8cube, dz=0.1)
    sc8cube.register_section(
        'test', section.StrikeSection(distance_idx=5))
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
        assert np.any(np.isnan(_data))  # check that some are False

    def test_dsv_get_display_arrays_stratigraphy(self):
        _data, _X, _Y = plot.get_display_arrays(self.dsv,
                                                data='stratigraphy')
        assert (_data.shape[0] == _X.shape[0] - 1)
        assert (_data.shape[1] == _Y.shape[1] - 1)
        assert (_data.shape[0] == _X.shape[0] - 1)
        assert (_data.shape[1] == _Y.shape[1] - 1)

    def test_dsv_get_display_arrays_badstring(self):
        with pytest.raises(ValueError, match=r'Bad data *.'):
            _data, _X, _Y = plot.get_display_arrays(
                self.dsv, data='badstring')

    def test_ssv_get_display_arrays_spacetime(self):
        with pytest.raises(AttributeError,
                           match=r'No "spacetime" or "preserved"*.'):
            _data, _X, _Y = plot.get_display_arrays(
                self.ssv, data='spacetime')

    def test_ssv_get_display_arrays_preserved(self):
        with pytest.raises(AttributeError,
                           match=r'No "spacetime" or "preserved"*.'):
            _data, _X, _Y = plot.get_display_arrays(self.ssv,
                                                    data='preserved')

    def test_ssv_get_display_arrays_stratigraphy(self):
        _data, _X, _Y = plot.get_display_arrays(
            self.ssv, data='stratigraphy')
        assert (_data.shape == _X.shape) and (_data.shape == _Y.shape)

    def test_ssv_get_display_arrays_badstring(self):
        with pytest.raises(ValueError, match=r'Bad data *.'):
            _data, _X, _Y = plot.get_display_arrays(
                self.ssv, data='badstring')


class TestGetDisplayLines:

    rcm8cube_nostrat = cube.DataCube(golf_path)
    rcm8cube_nostrat.register_section(
        'test', section.StrikeSection(distance_idx=5))
    dsv_nostrat = rcm8cube_nostrat.sections['test']['velocity']

    rcm8cube = cube.DataCube(golf_path)
    rcm8cube.stratigraphy_from('eta', dz=0.1)
    rcm8cube.register_section(
        'test', section.StrikeSection(distance_idx=5))
    dsv = rcm8cube.sections['test']['velocity']

    sc8cube = cube.StratigraphyCube.from_DataCube(rcm8cube, dz=0.1)
    sc8cube.register_section(
        'test', section.StrikeSection(distance_idx=5))
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

    def test_dsv_get_display_lines_badstring(self):
        with pytest.raises(ValueError, match=r'Bad data*.'):
            plot.get_display_lines(self.dsv,
                                   data='badstring')

    def test_ssv_get_display_lines_spacetime(self):
        with pytest.raises(AttributeError,
                           match=r'No "spacetime" or "preserved"*.'):
            _data, _segments = plot.get_display_lines(self.ssv,
                                                      data='spacetime')

    def test_ssv_get_display_lines_preserved(self):
        with pytest.raises(AttributeError,
                           match=r'No "spacetime" or "preserved"*.'):
            plot.get_display_lines(self.ssv,
                                   data='preserved')

    def test_ssv_get_display_lines_badstring(self):
        with pytest.raises(ValueError, match=r'Bad data*.'):
            plot.get_display_lines(self.ssv,
                                   data='badstring')

    @pytest.mark.xfail(raises=NotImplementedError, strict=True,
                       reason='Have not determined how to implement yet.')
    def test_ssv_get_display_lines_stratigraphy(self):
        plot.get_display_lines(self.ssv,
                               data='stratigraphy')


class TestGetDisplayLimits:

    rcm8cube_nostrat = cube.DataCube(golf_path)
    rcm8cube_nostrat.register_section(
        'test', section.StrikeSection(distance_idx=5))
    dsv_nostrat = rcm8cube_nostrat.sections['test']['velocity']

    rcm8cube = cube.DataCube(golf_path)
    rcm8cube.stratigraphy_from('eta', dz=0.1)
    rcm8cube.register_section(
        'test', section.StrikeSection(distance_idx=5))
    dsv = rcm8cube.sections['test']['velocity']

    sc8cube = cube.StratigraphyCube.from_DataCube(rcm8cube, dz=0.1)
    sc8cube.register_section(
        'test', section.StrikeSection(distance_idx=5))
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

    def test_dsv_get_display_limits_badstring(self):
        with pytest.raises(ValueError, match=r'Bad data*.'):
            plot.get_display_limits(self.dsv,
                                    data='badstring')

    def test_ssv_get_display_limits_spacetime(self):
        with pytest.raises(AttributeError,
                           match=r'No "spacetime" or "preserved"*.'):
            _ = plot.get_display_limits(self.ssv, data='spacetime')

    def test_ssv_get_display_limits_preserved(self):
        with pytest.raises(AttributeError,
                           match=r'No "spacetime" or "preserved"*.'):
            _ = plot.get_display_limits(self.ssv, data='preserved')

    def test_ssv_get_display_limits_stratigraphy(self):
        _lims = plot.get_display_limits(self.ssv, data='stratigraphy')
        assert len(_lims) == 4

    def test_ssv_get_display_limits_badstring(self):
        with pytest.raises(ValueError, match=r'Bad data*.'):
            plot.get_display_limits(self.ssv,
                                    data='badstring')


class TestColorMapFunctions:
    # note, no plotting, just boundaries and values checking

    rcm8cube = cube.DataCube(golf_path)

    def test_cartographic_SL0_defaults(self):
        H_SL = 0
        cmap, norm = plot.cartographic_colormap(H_SL)

        assert cmap.colors.shape == (10, 4)
        assert norm.boundaries[0] == H_SL-4.5
        assert norm.boundaries[-1] == H_SL+1

    def test_cartographic_SL0_h(self):
        H_SL = 0
        _h = 10
        cmap, norm = plot.cartographic_colormap(H_SL, h=_h)

        assert cmap.colors.shape == (10, 4)
        assert norm.boundaries[0] == H_SL-_h
        assert norm.boundaries[-1] == H_SL+1

    def test_cartographic_SL0_n(self):
        H_SL = 0
        _n = 0.1
        cmap, norm = plot.cartographic_colormap(H_SL, n=_n)

        assert cmap.colors.shape == (10, 4)
        assert norm.boundaries[0] == H_SL-4.5
        assert norm.boundaries[-1] == H_SL+(_n)

    def test_cartographic_SLn1_defaults(self):
        H_SL = -1
        cmap, norm = plot.cartographic_colormap(H_SL)

        assert cmap.colors.shape == (10, 4)
        assert norm.boundaries[0] == H_SL-4.5
        assert norm.boundaries[-1] == H_SL+1

    def test_cartographic_SL1_hn(self):
        H_SL = -1
        _h = 10
        _n = 0.1
        cmap, norm = plot.cartographic_colormap(H_SL, h=_h, n=_n)

        assert cmap.colors.shape == (10, 4)
        assert norm.boundaries[0] == H_SL-_h
        assert norm.boundaries[-1] == H_SL+(_n)


class TestScaleLightness:

    def test_no_scaling_one(self):
        _in = (0.12156862745098039, 0.4666666666666667, 0.7058823529411765)
        _out = plot._scale_lightness(_in, 1)
        assert _in == pytest.approx(_out)
        assert isinstance(_out, tuple)

    def test_limits_zero_one(self):
        red = (1.0, 0.0, 0.0)  # initial color red
        _zero = plot._scale_lightness(red, 0)
        _one = plot._scale_lightness(red, 1)
        _oneplus = plot._scale_lightness(red, 1.5)
        assert _zero[0] == pytest.approx(0)
        assert _one[0] == pytest.approx(1)
        assert _oneplus[0] == pytest.approx(1)

    def test_scale_down_red(self):
        red = (1.0, 0.0, 0.0)  # initial color red
        scales = np.arange(1, 0, -0.05)  # from 1 to 0.05
        for s, scale in enumerate(scales):
            darker_red = plot._scale_lightness(red, scale)
            assert darker_red[0] == pytest.approx(scale)


class TestShowHistograms:

    locs = [0.25, 1, 0.5, 4, 2]
    scales = [0.1, 0.25, 0.4, 0.5, 0.1]
    bins = np.linspace(0, 6, num=40)

    def test_one_histogram(self):
        _h, _b = np.histogram(
            np.random.normal(self.locs[3], self.scales[0], size=500),
            bins=self.bins, density=True)
        plot.show_histograms((_h, _b))
        plt.close()

    def test_one_histogram_with_ax(self):
        _h, _b = np.histogram(
            np.random.normal(self.locs[3], self.scales[0], size=500),
            bins=self.bins, density=True)
        fig, ax = plt.subplots()
        plot.show_histograms((_h, _b), ax=ax)
        plt.close()

    def test_multiple_no_sets(self):
        sets = [np.histogram(np.random.normal(lc, s, size=500),
                bins=self.bins, density=True) for lc, s in
                zip(self.locs, self.scales)]
        fig, ax = plt.subplots()
        plot.show_histograms(*sets, ax=ax)
        plt.close()

    def test_multiple_no_sets_alphakwarg(self):
        sets = [np.histogram(np.random.normal(lc, s, size=500),
                bins=self.bins, density=True) for lc, s in
                zip(self.locs, self.scales)]
        fig, ax = plt.subplots()
        plot.show_histograms(*sets, ax=ax, alpha=0.4)
        plt.close()

    def test_multiple_with_sets(self):
        sets = [np.histogram(np.random.normal(lc, s, size=500),
                bins=self.bins, density=True) for lc, s in
                zip(self.locs, self.scales)]
        fig, ax = plt.subplots()
        plot.show_histograms(*sets, sets=[0, 0, 1, 1, 2], ax=ax)
        plt.close()
        with pytest.raises(ValueError, match=r'Number of histogram tuples*.'):
            # input lengths must match
            plot.show_histograms(*sets, sets=[0, 1], ax=ax)
            plt.close()


class TestAerialView:
    """These are just "does it work" tests."""

    def test_makes_plot(self):
        _e = np.random.uniform(0, 1, size=(50, 50))
        fig, ax = plt.subplots()
        plot.aerial_view(
            _e, ax=ax)
        plt.close()

    def test_makes_plot_diffdatum(self):
        _e = np.random.uniform(0, 1, size=(50, 50))
        fig, ax = plt.subplots()
        plot.aerial_view(
            _e, datum=10, ax=ax)
        plt.close()

    def test_makes_plot_ticks(self):
        _e = np.random.uniform(0, 1, size=(50, 50))
        fig, ax = plt.subplots()
        plot.aerial_view(
            _e, ax=ax, ticks=True)
        plt.close()

    def test_makes_plot_colorbar_kw(self):
        _e = np.random.uniform(0, 1, size=(50, 50))
        fig, ax = plt.subplots()
        plot.aerial_view(
            _e, ax=ax, colorbar_kw={'labelsize': 6})
        plt.close()

    def test_makes_plot_no_ax(self):
        _e = np.random.uniform(0, 1, size=(50, 50))
        plot.aerial_view(
            _e)
        plt.close()


class TestOverlaySparseArray:
    """These are just "does it work" tests."""

    def test_makes_plot(self):
        _e = np.random.uniform(0, 73, size=(50, 50))
        fig, ax = plt.subplots()
        plot.overlay_sparse_array(_e, ax=ax)
        plt.close()

    def test_makes_plot_cmap_str(self):
        _e = np.random.uniform(0, 73, size=(50, 50))
        fig, ax = plt.subplots()
        plot.overlay_sparse_array(
            _e, cmap='Blues', ax=ax)
        plt.close()

    def test_makes_plot_cmap_obj(self):
        _e = np.random.uniform(0, 73, size=(50, 50))
        _cmap = plt.cm.get_cmap('Oranges', 64)
        fig, ax = plt.subplots()
        plot.overlay_sparse_array(
            _e, cmap=_cmap, ax=ax)
        plt.close()

    def test_makes_clips(self):
        _e = np.random.uniform(0, 73, size=(50, 50))
        fig, ax = plt.subplots()
        plot.overlay_sparse_array(
            _e, ax=ax,
            alpha_clip=(None, None))
        plot.overlay_sparse_array(
            _e, ax=ax,
            alpha_clip=(20, 30))
        plot.overlay_sparse_array(
            _e, ax=ax,
            alpha_clip=(50, None))
        with pytest.raises(TypeError, match=r'`alpha_clip` .* type .*'):
            plot.overlay_sparse_array(
                _e, ax=ax,
                alpha_clip=50)
        with pytest.raises(ValueError, match=r'`alpha_clip` .* length .*'):
            plot.overlay_sparse_array(
                _e, ax=ax,
                alpha_clip=(50, 90, 1000))
        plt.close()

    def test_clip_type_option(self):
        _e = np.random.uniform(0, 73, size=(50, 50))
        fig, ax = plt.subplots()
        plot.overlay_sparse_array(
            _e, ax=ax,
            alpha_clip=(None, None), clip_type='value')
        plot.overlay_sparse_array(
            _e, ax=ax,
            alpha_clip=(20, 30), clip_type='value')
        plot.overlay_sparse_array(
            _e, ax=ax,
            alpha_clip=(50, None), clip_type='value')
        plot.overlay_sparse_array(
            _e, ax=ax,
            alpha_clip=(None, None), clip_type='percentile')
        plot.overlay_sparse_array(
            _e, ax=ax,
            alpha_clip=(20, 30), clip_type='percentile')
        plot.overlay_sparse_array(
            _e, ax=ax,
            alpha_clip=(50, None), clip_type='percentile')
        with pytest.raises(ValueError, match=r'Bad value .*'):
            plot.overlay_sparse_array(
                _e, ax=ax,
                alpha_clip=(50, None), clip_type='invalidstring')
        with pytest.raises(ValueError, match=r'Bad value .*'):
            plot.overlay_sparse_array(
                _e, ax=ax,
                alpha_clip=(50, None), clip_type=(30, 30))
        plt.close()

    def test_makes_plot_no_ax(self):
        _e = np.random.uniform(0, 73, size=(50, 50))
        plot.overlay_sparse_array(
            _e)
        plt.close()
