import sys

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest

from sandplover.cube import DataCube
from sandplover.cube import StratigraphyCube
from sandplover.plot import _fill_steps
from sandplover.plot import _scale_lightness
from sandplover.plot import aerial_view
from sandplover.plot import append_colorbar
from sandplover.plot import cartographic_colormap
from sandplover.plot import get_display_arrays
from sandplover.plot import get_display_limits
from sandplover.plot import get_display_lines
from sandplover.plot import overlay_sparse_array
from sandplover.plot import show_histograms
from sandplover.plot import show_one_dimensional_trajectory_to_strata
from sandplover.plot import style_axes_km
from sandplover.plot import vintage_colormap
from sandplover.sample_data.sample_data import _get_golf_path
from sandplover.section import StrikeSection
from sandplover.utils import NoStratigraphyError

golf_path = _get_golf_path()


@pytest.mark.skipif(sys.platform == "win32", reason="TCL install error common.")
class TestAppendColorbar:
    def test_append_colorbar_working(self):
        """Test that the routine works.
        Doesn't really make any meaningful assertions.
        """
        _arr = np.random.randint(0, 100, size=(50, 50))
        fig, ax = plt.subplots()
        im = ax.imshow(_arr)
        cb = append_colorbar(im, ax)
        assert isinstance(cb, matplotlib.colorbar.Colorbar)
        assert ax.use_sticky_edges is False

    def test_size_argument_passed(self):
        """Test that the routine works.
        Doesn't really make any meaningful assertions.
        """
        _arr = np.random.randint(0, 100, size=(50, 50))
        fig, ax = plt.subplots()
        im = ax.imshow(_arr)
        cb = append_colorbar(im, ax, size=10)
        assert isinstance(cb, matplotlib.colorbar.Colorbar)

    def test_kwargs_argument_passed(self):
        _arr = np.random.randint(0, 100, size=(50, 50))
        fig, ax = plt.subplots()
        im = ax.imshow(_arr)
        _formatter = plt.FuncFormatter(lambda val, loc: np.round(val, 0))
        cb = append_colorbar(im, ax, size=10, format=_formatter)
        assert isinstance(cb, matplotlib.colorbar.Colorbar)
        assert cb.formatter is _formatter


@pytest.mark.skipif(sys.platform == "win32", reason="TCL install error common.")
class TestStyleAxesKm:
    def test_style_axes_km_ax(self):
        fig, ax = plt.subplots(1, 6)
        style_axes_km(ax[0])  # both
        style_axes_km(ax[1], "x")  # x only
        style_axes_km(ax[2], "y")  # y only
        style_axes_km(ax[3], "xy")  # both
        style_axes_km(ax[4], "z")  # do nothing
        ax[5].xaxis.set_major_formatter(style_axes_km)  # x only

        # check that the formatter has been set (or not)
        #   note, returns '' if formatter not changed
        assert ax[0].xaxis.get_major_formatter()(1000) == "1"
        assert ax[0].yaxis.get_major_formatter()(1000) == "1"

        assert ax[1].xaxis.get_major_formatter()(1000) == "1"
        assert ax[1].yaxis.get_major_formatter()(1000) == ""

        assert ax[2].xaxis.get_major_formatter()(1000) == ""
        assert ax[2].yaxis.get_major_formatter()(1000) == "1"

        assert ax[3].xaxis.get_major_formatter()(1000) == "1"
        assert ax[3].yaxis.get_major_formatter()(1000) == "1"

        assert ax[4].xaxis.get_major_formatter()(1000) == ""
        assert ax[4].yaxis.get_major_formatter()(1000) == ""

        assert ax[5].xaxis.get_major_formatter()(1000) == "1"
        assert ax[5].yaxis.get_major_formatter()(1000) == ""

        plt.close()


@pytest.mark.skipif(sys.platform == "win32", reason="TCL install error common.")
class TestFillSteps:
    """Test the `_fill_steps` function."""

    arr = np.array(
        [False, False, True, True, False, True, True, True, True, False, True]
    )

    def num_patches(self, pc):
        """hacky util to get length of PatchCollection."""
        return len(pc.properties()["facecolor"])

    def test_return_type(self):
        pc = _fill_steps(self.arr)
        assert isinstance(pc, matplotlib.collections.PatchCollection)

    def test_return_length_zero(self):
        _arr = np.array([False])
        pc = _fill_steps(_arr)
        assert self.num_patches(pc) == 0

    def test_return_length_zero_trues(self):
        _arr = np.array([True])
        pc = _fill_steps(_arr)
        assert self.num_patches(pc) == 0

    def test_return_length_one(self):
        _arr = np.array([False, True])
        pc = _fill_steps(_arr)
        assert self.num_patches(pc) == 1

    def test_return_length_three_get_two(self):
        _arr = np.array([False, True, True])
        pc = _fill_steps(_arr)
        assert self.num_patches(pc) == 2

    def test_return_length_three_get_two_trues(self):
        _arr = np.array([True, True, True])
        pc = _fill_steps(_arr)
        assert self.num_patches(pc) == 2

    def test_return_length_three_get_five(self):
        _arr = np.array(
            [False, True, True, False, False, False, True, True, False, True]
        )
        pc = _fill_steps(_arr)
        assert self.num_patches(pc) == 5

    def test_kwargs_default(self):
        pc = _fill_steps(self.arr)
        assert self.num_patches(pc) == 7
        _exp = pytest.approx(np.array([0.12156863, 0.46666667, 0.70588235, 1.0]))
        assert np.all(pc.get_facecolors()[0] == _exp)

    def test_kwargs_facecolor(self):
        pc = _fill_steps(self.arr, facecolor="green")
        assert self.num_patches(pc) == 7
        _exp = pytest.approx(np.array([0.0, 0.50196078, 0.0, 1.0]))
        assert np.all(pc.get_facecolors()[0] == _exp)


@pytest.mark.skipif(sys.platform == "win32", reason="TCL install error common.")
class TestSODTTST:
    """Test the `show_one_dimensional_trajectory_to_strata` function."""

    def test_sodttst_makes_plot(self):
        _e = np.random.randint(0, 10, size=(50,))
        fig, ax = plt.subplots()
        show_one_dimensional_trajectory_to_strata(_e, ax=ax, dz=0.1)
        plt.close()

    def test_sodttst_makes_labeled_strata(self):
        _e = np.random.randint(0, 10, size=(50,))
        fig, ax = plt.subplots()
        show_one_dimensional_trajectory_to_strata(_e, ax=ax, dz=0.1, label_strata=False)
        plt.close()

    def test_sodttst_makes_plot_lims_positives(self):
        _e = np.array([0, 1, 4, 5, 4, 10])
        fig, ax = plt.subplots()
        show_one_dimensional_trajectory_to_strata(_e, ax=ax, dz=0.1)
        assert ax.get_ylim() == (0, 12)
        plt.close()

    def test_sodttst_makes_plot_lims_negative(self):
        _e = np.array([10, -1, -4, -5, -4, -10])
        fig, ax = plt.subplots()
        show_one_dimensional_trajectory_to_strata(_e, ax=ax, dz=0.1)
        assert ax.get_ylim() == (-12, 12)
        plt.close()

    def test_sodttst_makes_plot_lims_negative_zero(self):
        _e = np.array([-1, -1, -4, -5, -4, -10])
        fig, ax = plt.subplots()
        show_one_dimensional_trajectory_to_strata(_e, ax=ax, dz=0.1)
        assert ax.get_ylim() == (-12, 0)
        plt.close()

    def test_sodttst_makes_plot_lims_equal(self):
        _e = np.array([-1, -1, -1, -1, -1, -1])
        fig, ax = plt.subplots()
        show_one_dimensional_trajectory_to_strata(_e, ax=ax, dz=0.1)
        assert ax.get_ylim() == (-1.2, 0)
        plt.close()

    def test_sodttst_makes_plot_sample_data(self):
        golfcube = DataCube(golf_path)
        locs = np.array(
            [
                [48, 152],
                [8, 63],
                [14, 102],
                [92, 118],
                [92, 168],
                [26, 114],
                [62, 135],
                [61, 171],
                [65, 193],
                [23, 175],
            ]
        )
        for i in range(10):
            _e = golfcube["eta"][:, locs[i, 0], locs[i, 1]]
            fig, ax = plt.subplots()
            show_one_dimensional_trajectory_to_strata(_e, ax=ax, dz=0.1)
            plt.close()

    def test_sodttst_makes_plot_no_ax(self):
        _e = np.random.randint(0, 10, size=(50,))
        show_one_dimensional_trajectory_to_strata(_e, dz=0.1)
        plt.close()

    def test_sodttst_makes_plot_3d_column(self):
        _e = np.random.randint(0, 10, size=(50, 1, 1))
        show_one_dimensional_trajectory_to_strata(_e, dz=0.1)
        plt.close()

    def test_sodttst_makes_plot_2d_column_error(self):
        _e = np.random.randint(0, 10, size=(50, 100, 1))
        with pytest.raises(ValueError, match=r'Elevation data "e" must *.'):
            show_one_dimensional_trajectory_to_strata(_e, dz=0.1)
        plt.close()

    def test_sodttst_makes_plot_subsidence_zeros(self):
        _e = np.array([0, 1, 4, 5, 4, 10])
        _s = np.array([0, 0, 0, 0, 0, 0])
        fig, ax = plt.subplots()
        show_one_dimensional_trajectory_to_strata(_e, sigma_dist=_s, ax=ax, dz=0.1)
        assert ax.get_ylim() == (0, 12)
        plt.close()

    def test_sodttst_makes_plot_lims_equal_via_subs(self):
        _e = np.arange(0, 6)
        fig, ax = plt.subplots()
        show_one_dimensional_trajectory_to_strata(_e, sigma_dist=1.0, ax=ax, dz=0.1)
        assert ax.get_ylim()[0] < 0
        assert ax.get_ylim()[1] > 0
        plt.close()


class TestGetDisplayArrays:
    golfcube_nostrat = DataCube(golf_path)
    golfcube_nostrat.register_section("test", StrikeSection(distance_idx=5))
    dsv_nostrat = golfcube_nostrat.sections["test"]["velocity"]

    golfcube = DataCube(golf_path)
    golfcube.stratigraphy_from("eta", dz=0.1)
    golfcube.register_section("test", StrikeSection(distance_idx=5))
    dsv = golfcube.sections["test"]["velocity"]

    scgolfcube = StratigraphyCube.from_DataCube(golfcube, dz=0.1)
    scgolfcube.register_section("test", StrikeSection(distance_idx=5))
    ssv = scgolfcube.sections["test"]["velocity"]

    def test_dsv_nostrat_get_display_arrays_spacetime(self):
        _data, _X, _Y = get_display_arrays(self.dsv_nostrat, data="spacetime")
        assert _data.shape[0] == _X.shape[0] - 1
        assert _data.shape[1] == _Y.shape[1] - 1
        assert _data.shape[0] == _X.shape[0] - 1
        assert _data.shape[1] == _Y.shape[1] - 1
        assert np.all(_data == self.dsv_nostrat)

    def test_dsv_nostrat_get_display_arrays_preserved(self):
        with pytest.raises(NoStratigraphyError):
            get_display_arrays(self.dsv_nostrat, data="preserved")

    def test_dsv_nostrat_get_display_arrays_stratigraphy(self):
        with pytest.raises(NoStratigraphyError):
            get_display_arrays(self.dsv_nostrat, data="stratigraphy")

    def test_dsv_get_display_arrays_spacetime(self):
        _data, _X, _Y = get_display_arrays(self.dsv, data="spacetime")
        assert _data.shape[0] == _X.shape[0] - 1
        assert _data.shape[1] == _Y.shape[1] - 1
        assert _data.shape[0] == _X.shape[0] - 1
        assert _data.shape[1] == _Y.shape[1] - 1
        assert np.all(_data == self.dsv)

    def test_dsv_get_display_arrays_preserved(self):
        _data, _X, _Y = get_display_arrays(self.dsv, data="preserved")
        assert _data.shape[0] == _X.shape[0] - 1
        assert _data.shape[1] == _Y.shape[1] - 1
        assert _data.shape[0] == _X.shape[0] - 1
        assert _data.shape[1] == _Y.shape[1] - 1
        assert np.any(np.isnan(_data))  # check that some are False

    def test_dsv_get_display_arrays_stratigraphy(self):
        _data, _X, _Y = get_display_arrays(self.dsv, data="stratigraphy")
        assert _data.shape[0] == _X.shape[0] - 1
        assert _data.shape[1] == _Y.shape[1] - 1
        assert _data.shape[0] == _X.shape[0] - 1
        assert _data.shape[1] == _Y.shape[1] - 1

    def test_dsv_get_display_arrays_badstring(self):
        with pytest.raises(ValueError, match=r"Bad data *."):
            _data, _X, _Y = get_display_arrays(self.dsv, data="badstring")

    def test_ssv_get_display_arrays_spacetime(self):
        with pytest.raises(AttributeError, match=r'No "spacetime" or "preserved"*.'):
            _data, _X, _Y = get_display_arrays(self.ssv, data="spacetime")

    def test_ssv_get_display_arrays_preserved(self):
        with pytest.raises(AttributeError, match=r'No "spacetime" or "preserved"*.'):
            _data, _X, _Y = get_display_arrays(self.ssv, data="preserved")

    def test_ssv_get_display_arrays_stratigraphy(self):
        _data, _X, _Y = get_display_arrays(self.ssv, data="stratigraphy")
        assert _data.shape[0] == _X.shape[0] - 1
        assert _data.shape[1] == _Y.shape[1] - 1
        assert _data.shape[0] == _X.shape[0] - 1
        assert _data.shape[1] == _Y.shape[1] - 1

    def test_ssv_get_display_arrays_badstring(self):
        with pytest.raises(ValueError, match=r"Bad data *."):
            _data, _X, _Y = get_display_arrays(self.ssv, data="badstring")


class TestGetDisplayLines:
    golfcube_nostrat = DataCube(golf_path)
    golfcube_nostrat.register_section("test", StrikeSection(distance_idx=5))
    dsv_nostrat = golfcube_nostrat.sections["test"]["velocity"]

    golfcube = DataCube(golf_path)
    golfcube.stratigraphy_from("eta", dz=0.1)
    golfcube.register_section("test", StrikeSection(distance_idx=5))
    dsv = golfcube.sections["test"]["velocity"]

    scgolfcube = StratigraphyCube.from_DataCube(golfcube, dz=0.1)
    scgolfcube.register_section("test", StrikeSection(distance_idx=5))
    ssv = scgolfcube.sections["test"]["velocity"]

    def test_dsv_nostrat_get_display_lines_spacetime(self):
        _data, _segments = get_display_lines(self.dsv_nostrat, data="spacetime")
        assert _segments.shape[1:] == (2, 2)

    def test_dsv_nostrat_get_display_lines_preserved(self):
        with pytest.raises(NoStratigraphyError):
            get_display_lines(self.dsv_nostrat, data="preserved")

    def test_dsv_nostrat_get_display_lines_stratigraphy(self):
        with pytest.raises(NoStratigraphyError):
            get_display_lines(self.dsv_nostrat, data="stratigraphy")

    def test_dsv_get_display_lines_spacetime(self):
        _data, _segments = get_display_lines(self.dsv, data="spacetime")
        assert _segments.shape[1:] == (2, 2)

    def test_dsv_get_display_lines_preserved(self):
        _data, _segments = get_display_lines(self.dsv, data="preserved")
        assert _segments.shape[1:] == (2, 2)

    def test_dsv_get_display_lines_stratigraphy(self):
        _data, _segments = get_display_lines(self.dsv, data="stratigraphy")
        assert _segments.shape[1:] == (2, 2)

    def test_dsv_get_display_lines_badstring(self):
        with pytest.raises(ValueError, match=r"Bad data*."):
            get_display_lines(self.dsv, data="badstring")

    def test_ssv_get_display_lines_spacetime(self):
        with pytest.raises(AttributeError, match=r'No "spacetime" or "preserved"*.'):
            _data, _segments = get_display_lines(self.ssv, data="spacetime")

    def test_ssv_get_display_lines_preserved(self):
        with pytest.raises(AttributeError, match=r'No "spacetime" or "preserved"*.'):
            get_display_lines(self.ssv, data="preserved")

    def test_ssv_get_display_lines_badstring(self):
        with pytest.raises(ValueError, match=r"Bad data*."):
            get_display_lines(self.ssv, data="badstring")

    @pytest.mark.xfail(
        raises=NotImplementedError,
        strict=True,
        reason="Have not determined how to implement yet.",
    )
    def test_ssv_get_display_lines_stratigraphy(self):
        get_display_lines(self.ssv, data="stratigraphy")


class TestGetDisplayLimits:
    golfcube_nostrat = DataCube(golf_path)
    golfcube_nostrat.register_section("test", StrikeSection(distance_idx=5))
    dsv_nostrat = golfcube_nostrat.sections["test"]["velocity"]

    golfcube = DataCube(golf_path)
    golfcube.stratigraphy_from("eta", dz=0.1)
    golfcube.register_section("test", StrikeSection(distance_idx=5))
    dsv = golfcube.sections["test"]["velocity"]

    scgolfcube = StratigraphyCube.from_DataCube(golfcube, dz=0.1)
    scgolfcube.register_section("test", StrikeSection(distance_idx=5))
    ssv = scgolfcube.sections["test"]["velocity"]

    def test_dsv_nostrat_get_display_limits_spacetime(self):
        _lims = get_display_limits(self.dsv_nostrat, data="spacetime")
        assert len(_lims) == 4

    def test_dsv_nostrat_get_display_limits_preserved(self):
        with pytest.raises(NoStratigraphyError):
            get_display_limits(self.dsv_nostrat, data="preserved")

    def test_dsv_nostrat_get_display_limits_stratigraphy(self):
        with pytest.raises(NoStratigraphyError):
            get_display_limits(self.dsv_nostrat, data="stratigraphy")

    def test_dsv_get_display_limits_spacetime(self):
        _lims = get_display_limits(self.dsv, data="spacetime")
        assert len(_lims) == 4

    def test_dsv_get_display_limits_preserved(self):
        _lims = get_display_limits(self.dsv, data="preserved")
        assert len(_lims) == 4

    def test_dsv_get_display_limits_stratigraphy(self):
        _lims = get_display_limits(self.dsv, data="stratigraphy")
        assert len(_lims) == 4

    def test_dsv_get_display_limits_badstring(self):
        with pytest.raises(ValueError, match=r"Bad data*."):
            get_display_limits(self.dsv, data="badstring")

    def test_ssv_get_display_limits_spacetime(self):
        with pytest.raises(AttributeError, match=r'No "spacetime" or "preserved"*.'):
            _ = get_display_limits(self.ssv, data="spacetime")

    def test_ssv_get_display_limits_preserved(self):
        with pytest.raises(AttributeError, match=r'No "spacetime" or "preserved"*.'):
            _ = get_display_limits(self.ssv, data="preserved")

    def test_ssv_get_display_limits_stratigraphy(self):
        _lims = get_display_limits(self.ssv, data="stratigraphy")
        assert len(_lims) == 4

    def test_ssv_get_display_limits_badstring(self):
        with pytest.raises(ValueError, match=r"Bad data*."):
            get_display_limits(self.ssv, data="badstring")


class TestColorMapFunctions:
    # note, no plotting, just boundaries and values checking

    golfcube = DataCube(golf_path)

    def test_cartographic_SL0_defaults(self):
        H_SL = 0
        cmap, norm = cartographic_colormap(H_SL)

        assert cmap.colors.shape == (10, 4)
        assert norm.boundaries[0] == H_SL - 4.5
        assert norm.boundaries[-1] == H_SL + 1

    def test_cartographic_SL0_h(self):
        H_SL = 0
        _h = 10
        cmap, norm = cartographic_colormap(H_SL, h=_h)

        assert cmap.colors.shape == (10, 4)
        assert norm.boundaries[0] == H_SL - _h
        assert norm.boundaries[-1] == H_SL + 1

    def test_cartographic_SL0_n(self):
        H_SL = 0
        _n = 0.1
        cmap, norm = cartographic_colormap(H_SL, n=_n)

        assert cmap.colors.shape == (10, 4)
        assert norm.boundaries[0] == H_SL - 4.5
        assert norm.boundaries[-1] == H_SL + (_n)

    def test_cartographic_SLn1_defaults(self):
        H_SL = -1
        cmap, norm = cartographic_colormap(H_SL)

        assert cmap.colors.shape == (10, 4)
        assert norm.boundaries[0] == H_SL - 4.5
        assert norm.boundaries[-1] == H_SL + 1

    def test_cartographic_SL1_hn(self):
        H_SL = -1
        _h = 10
        _n = 0.1
        cmap, norm = cartographic_colormap(H_SL, h=_h, n=_n)

        assert cmap.colors.shape == (10, 4)
        assert norm.boundaries[0] == H_SL - _h
        assert norm.boundaries[-1] == H_SL + (_n)

    def test_vintage_SL0_defaults(self):
        H_SL = 0
        cmap, norm = vintage_colormap(H_SL)

        assert cmap.colors.shape == (10, 3)
        assert norm.boundaries[0] == H_SL - 4.5
        assert norm.boundaries[-1] == H_SL + 1

    def test_vintage_SL0_h(self):
        H_SL = 0
        _h = 10
        cmap, norm = vintage_colormap(H_SL, h=_h)

        assert cmap.colors.shape == (10, 3)
        assert norm.boundaries[0] == H_SL - _h
        assert norm.boundaries[-1] == H_SL + 1

    def test_vintage_SL0_n(self):
        H_SL = 0
        _n = 0.1
        cmap, norm = vintage_colormap(H_SL, n=_n)

        assert cmap.colors.shape == (10, 3)
        assert norm.boundaries[0] == H_SL - 4.5
        assert norm.boundaries[-1] == H_SL + (_n)

    def test_vintage_SLn1_defaults(self):
        H_SL = -1
        cmap, norm = vintage_colormap(H_SL)

        assert cmap.colors.shape == (10, 3)
        assert norm.boundaries[0] == H_SL - 4.5
        assert norm.boundaries[-1] == 0  # default n=1 H_SL+1=0

    def test_vintage_SL1_hn(self):
        H_SL = -1
        _h = 10
        _n = 0.1
        cmap, norm = vintage_colormap(H_SL, h=_h, n=_n)

        assert cmap.colors.shape == (10, 3)
        assert norm.boundaries[0] == H_SL - _h
        assert norm.boundaries[-1] == H_SL + (_n)

    def test_vintage_pearson_recovered(self):
        # use pearsons bounds to recover the original
        cmap, norm = vintage_colormap(H_SL=0, h=20, n=10)

        assert cmap.colors.shape == (10, 3)
        assert norm.boundaries[0] == -20
        assert norm.boundaries[-1] == 10
        assert np.all(
            norm.boundaries == np.array([-20, -16, -12, -8, -5, -3, -1.2, 0, 1.4, 10])
        )


class TestScaleLightness:
    def test_no_scaling_one(self):
        _in = (0.12156862745098039, 0.4666666666666667, 0.7058823529411765)
        _out = _scale_lightness(_in, 1)
        assert _in == pytest.approx(_out)
        assert isinstance(_out, tuple)

    def test_limits_zero_one(self):
        red = (1.0, 0.0, 0.0)  # initial color red
        _zero = _scale_lightness(red, 0)
        _one = _scale_lightness(red, 1)
        _oneplus = _scale_lightness(red, 1.5)
        assert _zero[0] == pytest.approx(0)
        assert _one[0] == pytest.approx(1)
        assert _oneplus[0] == pytest.approx(1)

    def test_scale_down_red(self):
        red = (1.0, 0.0, 0.0)  # initial color red
        scales = np.arange(1, 0, -0.05)  # from 1 to 0.05
        for _, scale in enumerate(scales):
            darker_red = _scale_lightness(red, scale)
            assert darker_red[0] == pytest.approx(scale)


@pytest.mark.skipif(sys.platform == "win32", reason="TCL install error common.")
class TestShowHistograms:
    locs = [0.25, 1, 0.5, 4, 2]
    scales = [0.1, 0.25, 0.4, 0.5, 0.1]
    bins = np.linspace(0, 6, num=40)

    def test_one_histogram(self):
        _h, _b = np.histogram(
            np.random.normal(self.locs[3], self.scales[0], size=500),
            bins=self.bins,
            density=True,
        )
        show_histograms((_h, _b))
        plt.close()

    def test_one_histogram_with_ax(self):
        _h, _b = np.histogram(
            np.random.normal(self.locs[3], self.scales[0], size=500),
            bins=self.bins,
            density=True,
        )
        fig, ax = plt.subplots()
        show_histograms((_h, _b), ax=ax)
        plt.close()

    def test_multiple_no_sets(self):
        sets = [
            np.histogram(
                np.random.normal(lc, s, size=500), bins=self.bins, density=True
            )
            for lc, s in zip(self.locs, self.scales, strict=True)
        ]
        fig, ax = plt.subplots()
        show_histograms(*sets, ax=ax)
        plt.close()

    def test_multiple_no_sets_alphakwarg(self):
        sets = [
            np.histogram(
                np.random.normal(lc, s, size=500), bins=self.bins, density=True
            )
            for lc, s in zip(self.locs, self.scales, strict=True)
        ]
        fig, ax = plt.subplots()
        show_histograms(*sets, ax=ax, alpha=0.4)
        plt.close()

    def test_multiple_with_sets(self):
        sets = [
            np.histogram(
                np.random.normal(lc, s, size=500), bins=self.bins, density=True
            )
            for lc, s in zip(self.locs, self.scales, strict=True)
        ]
        fig, ax = plt.subplots()
        show_histograms(*sets, sets=[0, 0, 1, 1, 2], ax=ax)
        plt.close()
        with pytest.raises(ValueError, match=r"Number of histogram tuples*."):
            # input lengths must match
            show_histograms(*sets, sets=[0, 1], ax=ax)


@pytest.mark.skipif(sys.platform == "win32", reason="TCL install error common.")
class TestAerialView:
    """These are just "does it work" tests."""

    def test_makes_plot(self):
        _e = np.random.uniform(0, 1, size=(50, 50))
        fig, ax = plt.subplots()
        aerial_view(_e, ax=ax)
        plt.close()

    def test_makes_plot_diffdatum(self):
        _e = np.random.uniform(0, 1, size=(50, 50))
        fig, ax = plt.subplots()
        aerial_view(_e, datum=10, ax=ax)
        plt.close()

    def test_makes_plot_ticks(self):
        _e = np.random.uniform(0, 1, size=(50, 50))
        fig, ax = plt.subplots()
        aerial_view(_e, ax=ax, ticks=True)
        plt.close()

    def test_makes_plot_colorbar_kw(self):
        _e = np.random.uniform(0, 1, size=(50, 50))
        fig, ax = plt.subplots()
        aerial_view(_e, ax=ax, colorbar_kw={"labelsize": 6})
        plt.close()

    def test_makes_plot_no_ax(self):
        _e = np.random.uniform(0, 1, size=(50, 50))
        aerial_view(_e)
        plt.close()


@pytest.mark.skipif(sys.platform == "win32", reason="TCL install error common.")
class TestOverlaySparseArray:
    """These are just "does it work" tests."""

    def test_makes_plot(self):
        _e = np.random.uniform(0, 73, size=(50, 50))
        fig, ax = plt.subplots()
        overlay_sparse_array(_e, ax=ax)
        plt.close()

    def test_makes_plot_cmap_str(self):
        _e = np.random.uniform(0, 73, size=(50, 50))
        fig, ax = plt.subplots()
        overlay_sparse_array(_e, cmap="Blues", ax=ax)
        plt.close()

    def test_makes_plot_cmap_obj(self):
        _e = np.random.uniform(0, 73, size=(50, 50))
        _cmap = matplotlib.colormaps["Oranges"].resampled(64)
        fig, ax = plt.subplots()
        overlay_sparse_array(_e, cmap=_cmap, ax=ax)
        plt.close()

    def test_makes_clips(self):
        _e = np.random.uniform(0, 73, size=(50, 50))
        fig, ax = plt.subplots()
        overlay_sparse_array(_e, ax=ax, alpha_clip=(None, None))
        overlay_sparse_array(_e, ax=ax, alpha_clip=(20, 30))
        overlay_sparse_array(_e, ax=ax, alpha_clip=(50, None))
        with pytest.raises(TypeError, match=r"`alpha_clip` .* type .*"):
            overlay_sparse_array(_e, ax=ax, alpha_clip=50)
        with pytest.raises(ValueError, match=r"`alpha_clip` .* length .*"):
            overlay_sparse_array(_e, ax=ax, alpha_clip=(50, 90, 1000))
        plt.close()

    def test_clip_type_option(self):
        _e = np.random.uniform(0, 73, size=(50, 50))
        fig, ax = plt.subplots()
        overlay_sparse_array(_e, ax=ax, alpha_clip=(None, None), clip_type="value")
        overlay_sparse_array(_e, ax=ax, alpha_clip=(20, 30), clip_type="value")
        overlay_sparse_array(_e, ax=ax, alpha_clip=(50, None), clip_type="value")
        overlay_sparse_array(_e, ax=ax, alpha_clip=(None, None), clip_type="percentile")
        overlay_sparse_array(_e, ax=ax, alpha_clip=(20, 30), clip_type="percentile")
        overlay_sparse_array(_e, ax=ax, alpha_clip=(50, None), clip_type="percentile")
        with pytest.raises(ValueError, match=r"Bad value .*"):
            overlay_sparse_array(
                _e, ax=ax, alpha_clip=(50, None), clip_type="invalidstring"
            )
        with pytest.raises(ValueError, match=r"Bad value .*"):
            overlay_sparse_array(_e, ax=ax, alpha_clip=(50, None), clip_type=(30, 30))
        plt.close()

    def test_makes_plot_no_ax(self):
        _e = np.random.uniform(0, 73, size=(50, 50))
        overlay_sparse_array(_e)
        plt.close()
