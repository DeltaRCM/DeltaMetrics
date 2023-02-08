"""Tests for the mask.py script."""
import pytest

import numpy as np
import xarray as xr

from deltametrics import cube
from deltametrics import strat
from deltametrics.sample_data import _get_golf_path


golf_path = _get_golf_path()
golfcube = cube.DataCube(golf_path)


class TestComputeBoxyStratigraphyVolume:

    elev = golfcube['eta']
    time = golfcube['time']

    def test_returns_volume_and_elevations_given_dz(self):
        s, e = strat.compute_boxy_stratigraphy_volume(
            self.elev, self.time, dz=0.05)
        assert s.ndim == 3
        assert s.shape == e.shape
        assert e[1, 0, 0] - e[0, 0, 0] == pytest.approx(0.05)

    def test_returns_volume_and_elevations_given_z(self):
        z = np.linspace(-20, 500, 200)
        s, e = strat.compute_boxy_stratigraphy_volume(
            self.elev, self.time, z=z)
        assert s.ndim == 3
        assert s.shape == e.shape
        assert np.all(e[:, 0, 0] == z)

    def test_returns_volume_and_elevations_given_nz(self):
        s, e = strat.compute_boxy_stratigraphy_volume(
            self.elev, self.time, nz=33)
        assert s.ndim == 3
        assert s.shape == e.shape
        assert s.shape[0] == 33 + 1

    def test_returns_volume_and_elevations_given_subsidence(self):
        s, e = strat.compute_boxy_stratigraphy_volume(
            self.elev, self.time, sigma_dist=1, nz=33)
        assert s.ndim == 3
        assert s.shape == e.shape
        assert s.shape[0] == 33 + 1        

    @pytest.mark.xfail(raises=NotImplementedError,
                       strict=True, reason='Not yet developed.')
    def test_return_cube(self):
        s, e = strat.compute_boxy_stratigraphy_volume(
            self.elev, self.time,
            dz=0.05, return_cube=True)

    def test_lessthan3d_error(self):
        with pytest.raises(ValueError,
                           match=r'Input arrays must be three-dimensional.'):
            strat.compute_boxy_stratigraphy_volume(
                self.elev[:, 10, 120].squeeze(),
                self.time[:, 10, 120].squeeze(),
                dz=0.05)
        with pytest.raises(ValueError,
                           match=r'Input arrays must be three-dimensional.'):
            strat.compute_boxy_stratigraphy_volume(
                self.elev[:, 10, :].squeeze(),
                self.time[:, 10, :].squeeze(),
                dz=0.05)

    def test_bad_shape_error(self):
        with pytest.raises(ValueError,
                           match=r'Input arrays must be three-dimensional.'):
            strat.compute_boxy_stratigraphy_volume(
                self.elev[:, 10, 120].squeeze(),
                self.time[:, 10, 120].squeeze(),
                dz=0.05)

    def test_mismatch_shape_error(self):
        with pytest.raises(ValueError,
                           match=r'Mismatched input shapes "elev" and "prop".'):
            strat.compute_boxy_stratigraphy_volume(
                self.elev[:, 10:12, 120].squeeze(),
                self.time[:, 10, 120].squeeze(),
                dz=0.05)

    def test_no_z_options(self):
        with pytest.warns(UserWarning, match=r'No specification .*'):
            strat.compute_boxy_stratigraphy_volume(self.elev, self.time)


class TestComputeBoxyStratigraphyCoordinates:

    elev = golfcube['eta']
    time = golfcube['time']

    def test_returns_sc_dc_given_dz(self):
        # check for the warning when no option passed
        with pytest.warns(UserWarning):
            sc, dc = strat.compute_boxy_stratigraphy_coordinates(
                self.elev)
        # now specify dz
        sc, dc = strat.compute_boxy_stratigraphy_coordinates(
            self.elev, dz=0.05)
        assert sc.shape == dc.shape
        assert sc.shape[1] == 3
        assert sc.shape[0] > 1  # don't know how big it will be

    def test_returns_sc_dc(self):
        sc, dc = strat.compute_boxy_stratigraphy_coordinates(
            self.elev, dz=0.05)
        assert sc.shape == dc.shape
        assert sc.shape[1] == 3

    def test_returns_sc_dc_return_strata(self):
        sc, dc, s = strat.compute_boxy_stratigraphy_coordinates(
            self.elev, dz=0.05, return_strata=True)
        assert s.ndim == 3
        assert s.shape == self.elev.shape
        assert sc.shape == dc.shape
        assert sc.shape[1] == 3

    def test_returns_sc_dc_given_z(self):
        z = np.linspace(0, 0.25, 7)
        sc, dc = strat.compute_boxy_stratigraphy_coordinates(self.elev, z=z)
        assert np.min(sc[:, 0]) == 0
        assert np.max(sc[:, 0]) == 6
        # check that the number of z values matches len(z)
        assert np.unique(sc[:, 0]).shape[0] == len(z)

    def test_returns_sc_dc_given_nz(self):
        sc, dc = strat.compute_boxy_stratigraphy_coordinates(self.elev, nz=13)
        assert np.min(sc[:, 0]) == 0
        # check that the number of z values matches nz
        assert np.unique(sc[:, 0]).shape[0] == 13

    def test_returns_sc_dc_given_subsidence(self):
        sc, dc = strat.compute_boxy_stratigraphy_coordinates(
            self.elev, sigma_dist=1, nz=13)
        assert np.min(sc[:, 0]) == 0        

    @pytest.mark.xfail(raises=NotImplementedError,
                       strict=True, reason='Not yet developed.')
    def test_return_cube(self):
        s, sc, dc = strat.compute_boxy_stratigraphy_coordinates(
            self.elev, dz=0.05, return_cube=True)

    def test_no_z_options(self):
        with pytest.warns(UserWarning, match=r'No specification .*'):
            strat.compute_boxy_stratigraphy_coordinates(
                self.elev[:, 10, 120].squeeze())


class TestComputeElevationToPreservation:

    def test_1d_shorts(self):
        s1, p1 = strat._compute_elevation_to_preservation(np.array([1]))
        s15, p15 = strat._compute_elevation_to_preservation(np.array([5]))
        s2, p2 = strat._compute_elevation_to_preservation(np.array([1, 2]))
        s3, p3 = strat._compute_elevation_to_preservation(np.array([1, 2, 3]))
        assert np.all(s1 == np.array([1]))
        assert np.all(s15 == np.array([5]))
        assert np.all(p1 == np.array([False]))
        assert np.all(s2 == np.array([1, 2]))
        assert np.all(p2 == np.array([True, True]))
        assert np.all(s3 == np.array([1, 2, 3]))
        assert np.all(p3 == np.array([True, True, True]))

    def test_1d_all_zeros(self):
        s, p = strat._compute_elevation_to_preservation(
            np.array([0, 0, 0, 0]))
        assert np.all(s == np.array([0, 0, 0, 0]))
        assert np.all(p == np.array([False, False, False, False]))

    def test_1d_all_ones(self):
        s, p = strat._compute_elevation_to_preservation(
            np.array([1, 1, 1, 1]))
        assert np.all(s == np.array([1, 1, 1, 1]))
        assert np.all(p == np.array([False, False, False, False]))

    def test_1d_all_aggrade(self):
        s, p = strat._compute_elevation_to_preservation(
            np.array([0, 1, 2, 3]))
        assert np.all(s == np.array([0, 1, 2, 3]))
        assert np.all(p == np.array([True, True, True, True]))
        assert np.all(s[1:] - s[:-1] == 1)

    def test_1d_all_erode_positive(self):
        s, p = strat._compute_elevation_to_preservation(
            np.array([3, 2, 1, 0]))
        assert np.all(s == np.array([0, 0, 0, 0]))
        assert np.all(p == np.array([False, False, False, False]))

    def test_1d_all_erode_negative(self):
        s, p = strat._compute_elevation_to_preservation(
            np.array([0, -1, -2, -3]))
        assert np.all(s == np.array([-3, -3, -3, -3]))
        assert np.all(p == np.array([False, False, False, False]))

    def test_1d_up_down(self):
        s, p = strat._compute_elevation_to_preservation(
            np.array([0, 1, 2, 1]))
        assert np.all(s == np.array([0, 1, 1, 1]))
        assert np.all(p == np.array([True, True, False, False]))

    def test_1d_up_down_flat(self):
        s, p = strat._compute_elevation_to_preservation(
            np.array([0, 1, 2, 1, 1]))
        assert np.all(s == np.array([0, 1, 1, 1, 1]))
        assert np.all(p == np.array([True, True, False, False, False]))

    def test_1d_up_down_up(self):
        s, p = strat._compute_elevation_to_preservation(
            np.array([0, 1, 2, 1, 2]))
        assert np.all(s == np.array([0, 1, 1, 1, 2]))
        assert np.all(p == np.array([True, True, False, False, True]))

    def test_1d_up_down_down(self):
        s, p = strat._compute_elevation_to_preservation(
            np.array([0, 1, 2, 1, 0]))
        assert np.all(s == np.array([0, 0, 0, 0, 0]))
        assert np.all(p == np.array([False, False, False, False, False]))

    def test_2d_all_zeros(self):
        s, p = strat._compute_elevation_to_preservation(
            np.zeros((6, 4)))
        assert np.all(s == np.zeros((6, 4)))
        assert np.all(p == np.zeros((6, 4), dtype=bool))

    def test_2d_all_aggrade(self):
        e = np.tile(np.arange(0, 3), (2, 1)).T
        s, p = strat._compute_elevation_to_preservation(
            e)
        assert np.all(s == np.array([[0, 0], [1, 1], [2, 2]]))
        assert np.all(p == np.ones((3, 2), dtype=bool))

    def test_2d_different_walks(self):
        e = np.array([[0, 3,   4],
                      [1, 3,   3],
                      [1, 4,   4],
                      [2, 4.5, 3],
                      [3, 5,   4.5]])
        s, p = strat._compute_elevation_to_preservation(e)
        assert np.all(s == np.array([[0, 3,   3],
                                     [1, 3,   3],
                                     [1, 4,   3],
                                     [2, 4.5, 3],
                                     [3, 5,   4.5]]))
        assert np.all(p == np.array([[True,  False, False],
                                     [True,  False, False],
                                     [False, True,  False],
                                     [True,  True,  False],
                                     [True,  True,  True]]))

    def test_3d_all_zeros(self):
        s, p = strat._compute_elevation_to_preservation(np.zeros((6, 4, 4)))
        assert np.all(s == np.zeros((6, 4, 4)))
        assert np.all(p == np.zeros((6, 4, 4), dtype=bool))

    def test_3d_all_aggrade(self):
        e = np.tile(np.arange(0, 3), (2, 2, 1)).T
        s, p = strat._compute_elevation_to_preservation(e)
        assert np.all(s == np.array([[[0, 0], [0, 0]],
                                     [[1, 1], [1, 1]],
                                     [[2, 2], [2, 2]]]))
        assert np.all(p == np.ones((3, 2, 2), dtype=bool))

    def test_3d_different_walks_return_valid_only_check(self):
        e = np.random.rand(51, 120, 240)
        s, p = strat._compute_elevation_to_preservation(e)
        assert s.shape == (51, 120, 240)
        assert np.all(s[-1, ...] == e[-1, ...])
        assert np.all(s[0, ...] == np.min(e, axis=0))


class TestComputePreservationToCube:

    def test_1d_shorts(self):
        z = np.arange(0, 5, step=0.25)
        sc1, dc1 = strat._compute_preservation_to_cube(np.array([1]), z)
        sc15, dc15 = strat._compute_preservation_to_cube(np.array([5]), z)
        sc2, dc2 = strat._compute_preservation_to_cube(np.array([1, 2]), z)
        sc3, dc3 = strat._compute_preservation_to_cube(np.array([1, 2, 3]), z)
        # assert np.all(sc1 == np.array([3, 2, 1, 0]))


class TestOneDimStratigraphyExamples:
    """Tests for various cases of 1d stratigraphy."""

    def take_var_time(self, s, z, sc, dc):
        """Utility for testing, does the same workflow as the __getitem__ for
        section and plan variables.

        Parameters: s = strata, z = vert coordinate
        """
        t = np.arange(s.shape[0])  # x-axis time array
        c = np.full((z.shape[0]), np.nan)
        c[sc[:, 0]] = t[dc[:, 0]]
        return c

    def test_onedim_traj_drop_at_end(self):
        e = np.array([0, 1, 2, 3, 1])
        z = strat._determine_strat_coordinates(e, dz=0.5)  # vert coordinates
        assert z[0] == 0
        assert z[-1] == 3
        assert len(z) == 7
        s, p = strat._compute_elevation_to_preservation(e)
        sc, dc = strat._compute_preservation_to_cube(s, z)
        lst = np.argmin(s[:, ...] < s[-1, ...], axis=0)  # last elevation idx
        c = self.take_var_time(s, z, sc, dc)
        assert s[-1] == e[-1]
        assert s[-1] == e[-1]
        assert np.all(np.isnan(c[2:]))  # from z>=1 to z==3
        assert np.all(c[:2] == 1)
        assert lst == 1

    def test_onedim_traj_drop_at_end_to_zero(self):
        e = np.array([0, 1, 1, 0])
        # e = np.expand_dims(e, axis=(1,2))
        z = strat._determine_strat_coordinates(e, dz=0.5)  # vert coordinates
        s, p = strat._compute_elevation_to_preservation(e)
        sc, dc = strat._compute_preservation_to_cube(s, z)
        c = self.take_var_time(s, z, sc, dc)
        assert len(z) == 3
        assert np.all(np.isnan(c[:]))
        assert np.all(p == np.array([False, False, False, False]))

    def test_onedim_traj_upsanddowns(self):
        e = np.array([0, 0, 1, 4, 6, 5, 3.5, 5, 7, 5, 6])
        # e = np.expand_dims(e, axis=(1,2))
        z = strat._determine_strat_coordinates(e, dz=0.5)  # vert coordinates
        s, p = strat._compute_elevation_to_preservation(e)
        sc, dc = strat._compute_preservation_to_cube(s, z)
        c = self.take_var_time(s, z, sc, dc)
        assert z[-1] == 7
        assert s[-1] == 6
        assert np.all(p.nonzero()[0] == (2, 3, 7, 10))
        assert c[0] == 2

    def test_onedim_traj_upsanddowns_negatives(self):
        # e = np.array([0, 0, -1, -4, -2, 3, 3.5, 3, 3, 4, 4])
        e = xr.DataArray([0, 0, -1, -4, -2, 3, 3.5, 3, 3, 4, 4])
        z = strat._determine_strat_coordinates(e, dz=0.5)  # vert coordinates
        s, p = strat._compute_elevation_to_preservation(e)
        sc, dc = strat._compute_preservation_to_cube(s, z)
        c = self.take_var_time(s, z, sc, dc)
        assert np.all(p.nonzero()[0] == (4, 5, 9))


class TestDetermineStratCoordinates:

    def test_given_none_chooses_default(self):
        e = np.array([0, 1, 1, 2, 1])
        with pytest.warns(UserWarning, match=r'No specification *.'):
            _ = strat._determine_strat_coordinates(e)

    def test_given_z(self):
        e = np.array([0, 1, 1, 2, 1])
        z_in = np.arange(0, 10, step=0.2)
        z = strat._determine_strat_coordinates(e, z=z_in)
        assert np.all(z == z_in)

    def test_given_z_scalar(self):
        e = np.array([0, 1, 1, 2, 1])
        with pytest.raises(ValueError):
            _ = strat._determine_strat_coordinates(e, z=0.05)

    def test_given_dz(self):
        e = np.array([0, 1, 1, 2, 1])
        z = strat._determine_strat_coordinates(e, dz=0.05)
        assert len(z) == 41
        assert z[-1] == 2.00
        assert z[0] == 0
        assert z[1] - z[0] == 0.05

    def test_given_dz_negative_endpoint(self):
        e = np.array([0, 1, 1, 50, -1])
        z = strat._determine_strat_coordinates(e, dz=0.05)
        assert len(z) == 1021
        assert z[-1] == pytest.approx(50)
        assert z[0] == -1

    def test_given_dz_zero(self):
        e = np.array([0, 1, 1, 2, 1])
        with pytest.raises(ValueError, match=r'"dz" or "nz" cannot *.'):
            _ = strat._determine_strat_coordinates(e, dz=0)

    def test_given_dz_negative(self):
        e = np.array([0, 1, 1, 2, 1])
        with pytest.raises(ValueError, match=r'"dz" or "nz" cannot *.'):
            _ = strat._determine_strat_coordinates(e, dz=-0.5)

    def test_given_nz(self):
        e = np.array([0, 1, 1, 2, 1])
        z = strat._determine_strat_coordinates(e, nz=50)
        assert len(z) == 50 + 1
        assert z[-1] == 2.00
        assert z[0] == 0
        assert z[1] - z[0] == pytest.approx(2 / 50)  # delta_Z / nz

    def test_given_nz_negative_endpoint(self):
        e = np.array([0, 1, 1, 50, -1])
        z = strat._determine_strat_coordinates(e, nz=50)
        assert len(z) == 50 + 1
        assert z[-1] == pytest.approx(50)
        assert z[0] == -1
        assert z[1] - z[0] == pytest.approx(51 / 50)  # delta_Z / nz

    def test_given_nz_zero(self):
        e = np.array([0, 1, 1, 2, 1])
        with pytest.raises(ValueError, match=r'"dz" or "nz" cannot *.'):
            _ = strat._determine_strat_coordinates(e, nz=0)

    def test_given_nz_negative(self):
        e = np.array([0, 1, 1, 2, 1])
        with pytest.raises(ValueError, match=r'"dz" or "nz" cannot *.'):
            _ = strat._determine_strat_coordinates(e, nz=-5)

    def test_given_z_and_dz(self):
        e = np.array([0, 1, 1, 2, 1])
        z_in = np.arange(0, 10, step=0.2)
        z = strat._determine_strat_coordinates(e, z=z_in, dz=0.1)
        assert np.all(z == z_in)
        assert z[1] - z[0] == 0.2

    def test_given_z_and_nz(self):
        e = np.array([0, 1, 1, 2, 1])
        z_in = np.arange(0, 10, step=0.2)
        z = strat._determine_strat_coordinates(e, z=z_in, nz=0.1)
        assert np.all(z == z_in)
        assert z[1] - z[0] == 0.2

    def test_given_z_and_dz_and_nz(self):
        e = np.array([0, 1, 1, 2, 1])
        z_in = np.arange(0, 10, step=0.2)
        z = strat._determine_strat_coordinates(e, z=z_in, dz=0.1, nz=50000)
        assert np.all(z == z_in)
        assert z[1] - z[0] == 0.2

    def test_given_dz_and_nz(self):
        e = np.array([0, 1, 1, 2, 1])
        z = strat._determine_strat_coordinates(e, dz=0.1, nz=50000)
        assert z[0] == 0
        assert z[-1] == 2
        assert z[1] - z[0] == 0.1


class TestSubsidenceElevationAdjustment:
    
    def test_shapes_not_matching(self):
        e = np.zeros((5, 2, 1))
        s = np.zeros((2, 4, 2))
        with pytest.raises(ValueError):
            strat._adjust_elevation_by_subsidence(e, s)

    def test_1D_sigma_dist_as_float(self):
        e = np.zeros((10,))
        s = 1.0
        adj = strat._adjust_elevation_by_subsidence(e, s)
        assert np.all(adj == np.arange(-9, 1))

    def test_1D_sigma_dist_as_int(self):
        e = np.zeros((10,))
        s = 1
        adj = strat._adjust_elevation_by_subsidence(e, s)
        assert np.all(adj == np.arange(-9, 1))

    def test_2d_sigma_dist_3d_elev(self):
        e = np.zeros((5, 2, 3))
        s = np.ones((2, 3))
        adj = strat._adjust_elevation_by_subsidence(e, s)
        assert adj.shape == e.shape
        assert adj[0, 0, 0] == -4
        assert adj[-1, 0, 0] == 0.0

    def test_1d_sigma_dist_3d_elev(self):
        e = np.zeros((5, 2, 3))
        s = np.ones((5,))
        adj = strat._adjust_elevation_by_subsidence(e, s)
        assert adj.shape == e.shape
        assert adj[0, 0, 0] == -1.0
        assert adj[-1, 0, 0] == -1.0

    def test_1d_flat(self):
        topo = np.array([0, 0, 0, 0, 0, 0])  # recorded as eta
        sigma_dist = np.array([0, 1, 2, 3, 4, 5])  # known subsidence
        # apply function
        adj = strat._adjust_elevation_by_subsidence(topo, sigma_dist)
        # adjusted elevations, lowest (first) should be -5
        # or total subsided distance, while final should match present elev
        assert adj[0] == -1 * sigma_dist[-1]
        assert adj[-1] == topo[-1]

    def test_1d_with_uplift(self):
        topo = np.array([0, 0, 2, 3, 4])
        sigma_dist = np.array([0, -2, -2, -1, -1])
        # apply function
        adj = strat._adjust_elevation_by_subsidence(topo, sigma_dist)
        # bottom of strat column should be at an elevation of 0
        assert adj[0] == 0.0
        # final value should equal combo of topo and subsidence at end
        assert adj[-1] == topo[-1] + sigma_dist[-1]


class TestComputeNetToGross:

    golfstrat = cube.StratigraphyCube.from_DataCube(golfcube, dz=0.1)

    def test_net_to_gross_nobg(self):
        net_to_gross = strat.compute_net_to_gross(
            self.golfstrat['sandfrac'],
            net_threshold=0.5,
            background=None)
        assert np.all(net_to_gross) <= 1
        assert np.all(net_to_gross) >= 0

    def test_net_to_gross(self):
        background = (self.golfstrat.Z > np.min(golfcube['eta'].data, axis=0))
        net_to_gross = strat.compute_net_to_gross(
            self.golfstrat['sandfrac'],
            net_threshold=0.5,
            background=background)
        assert np.all(net_to_gross) <= 1
        assert np.all(net_to_gross) >= 0

    def test_net_to_gross_thresh0(self):
        background = (self.golfstrat.Z > np.min(golfcube['eta'].data, axis=0))
        net_to_gross = strat.compute_net_to_gross(
            self.golfstrat['sandfrac'],
            net_threshold=0.01,
            background=background)
        assert np.all(net_to_gross) <= 1
        assert np.all(net_to_gross) >= 0

    def test_net_to_gross_nothresh_default_is_half(self):
        background = (self.golfstrat.Z > np.min(golfcube['eta'].data, axis=0))
        net_to_gross_05 = strat.compute_net_to_gross(
            self.golfstrat['sandfrac'],
            net_threshold=0.5,
            background=background)
        net_to_gross_def = strat.compute_net_to_gross(
            self.golfstrat['sandfrac'],
            background=background)
        assert np.all(net_to_gross_def) <= 1
        assert np.all(net_to_gross_def) >= 0
        assert np.all(
            net_to_gross_def[~np.isnan(net_to_gross_def)]
            == net_to_gross_05[~np.isnan(net_to_gross_05)])


class TestComputeThicknessSurfaces:

    def test_compute_thickness_0(self):
        deposit_thickness0 = strat.compute_thickness_surfaces(
            golfcube['eta'][0, :, :],
            golfcube['eta'][0, :, :])
        zeros = (deposit_thickness0 == 0)
        nans = np.isnan(deposit_thickness0)
        assert np.all(np.logical_or(zeros, nans))  # all 0 or nan

    def test_compute_thickness_1(self):
        deposit_thickness1 = strat.compute_thickness_surfaces(
            golfcube['eta'][0, :, :],
            golfcube['eta'][1, :, :])
        zeros = (deposit_thickness1 == 0)
        nans = np.isnan(deposit_thickness1)
        assert np.any(~np.logical_or(zeros, nans))  # any not nan or 0

    def test_compute_thickness_total(self):
        deposit_thickness = strat.compute_thickness_surfaces(
            golfcube['eta'][-1, :, :],
            np.min(golfcube['eta'], axis=0))
        # zeros = (deposit_thickness == 0)
        gtr_hb = (deposit_thickness > golfcube.meta['hb'].data)
        # nans = np.isnan(deposit_thickness)
        assert np.any(gtr_hb) # any greater than thickness


class TestComputeSedimentograph:

    golfstrat = cube.StratigraphyCube.from_DataCube(golfcube, dz=0.1)

    def test_two_bins(self):
        background = (self.golfstrat.Z > np.min(golfcube['eta'].data, axis=0))
        (s, r, b) = strat.compute_sedimentograph(
            self.golfstrat['sandfrac'],
            background=background)
        assert np.all(np.logical_or(s <= 1, np.isnan(s)))
        assert s.shape[0] == 10  # default is 10 sections
        assert s.shape[1] == 2  # default is 2 bins
        assert b.shape[0] == 3  # default is 2 bins, 3 edges
        assert r.shape[0] == s.shape[0]
        assert b.shape[0] - 1 == s.shape[1]  # edges - 1 is shape of sedgraph

    def test_two_bins_with_origin(self):
        background = (self.golfstrat.Z > np.min(golfcube['eta'].data, axis=0))
        (s, r, b) = strat.compute_sedimentograph(
            self.golfstrat['sandfrac'],
            background=background,
            origin_idx=[3, 100])
        assert np.all(np.logical_or(s <= 1, np.isnan(s)))
        assert s.shape[0] == 10  # default is 10 sections
        assert r.shape[0] == s.shape[0]
        assert b.shape[0] - 1 == s.shape[1]  # edges - 1 is shape of sedgraph

    def test_two_bins_more_sects(self):
        background = (self.golfstrat.Z > np.min(golfcube['eta'].data, axis=0))
        (s, r, b) = strat.compute_sedimentograph(
            self.golfstrat['sandfrac'],
            num_sections=50,
            background=background,
            origin_idx=[3, 100])
        assert np.all(np.logical_or(s <= 1, np.isnan(s)))
        assert s.shape[0] == 50  # match input
        assert r.shape[0] == s.shape[0]
        assert b.shape[0] - 1 == s.shape[1]  # edges - 1 is shape of sedgraph

    def test_two_bins_cust_rad(self):
        background = (self.golfstrat.Z > np.min(golfcube['eta'].data, axis=0))
        (s, r, b) = strat.compute_sedimentograph(
            self.golfstrat['sandfrac'],
            last_section_radius=2750,
            background=background,
            origin_idx=[3, 100])
        assert np.all(np.logical_or(s <= 1, np.isnan(s)))
        assert r.shape[0] == s.shape[0]
        assert b.shape[0] - 1 == s.shape[1]  # edges - 1 is shape of sedgraph
        assert r[-1] == 2750

    def test_two_bins_cust_rad_long(self):
        background = (self.golfstrat.Z > np.min(golfcube['eta'].data, axis=0))
        (s, r, b) = strat.compute_sedimentograph(
            self.golfstrat['sandfrac'],
            last_section_radius=4000,
            background=background,
            origin_idx=[3, 100])
        assert np.all(np.logical_or(s <= 1, np.isnan(s)))
        assert r.shape[0] == s.shape[0]
        assert b.shape[0] - 1 == s.shape[1]  # edges - 1 is shape of sedgraph
        assert r[-1] == 4000
        assert np.any(np.isnan(s))# should be some nan

    def test_five_bins(self):
        background = (self.golfstrat.Z > np.min(golfcube['eta'].data, axis=0))
        (s, r, b) = strat.compute_sedimentograph(
            self.golfstrat['sandfrac'],
            sediment_bins=np.linspace(0, 1, num=6, endpoint=True),
            background=background,
            origin_idx=[3, 100])
        assert np.all(np.logical_or(s <= 1, np.isnan(s)))
        assert s.shape[1] == 5  # default is 2 bins
        assert b.shape[0] == 6  # default is 2 bins, 3 edges
        assert r.shape[0] == s.shape[0]
        assert b.shape[0] - 1 == s.shape[1]  # edges - 1 is shape of sedgraph

    def test_time_variable(self):
        background = (self.golfstrat.Z > np.min(golfcube['eta'].data, axis=0))
        
        (s, r, b) = strat.compute_sedimentograph(
            self.golfstrat['time'],
            num_sections=50,
            last_section_radius=2750,
            sediment_bins=np.linspace(0, golfcube.t[-1], num=5),
            background=background,
            origin_idx=[3, 100])
        assert np.all(np.logical_or(s <= 1, np.isnan(s)))
