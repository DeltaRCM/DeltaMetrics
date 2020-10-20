"""Tests for the mask.py script."""
import pytest
import sys
import os
import numpy as np
import xarray as xr

from deltametrics import cube
from deltametrics import strat

# initialize a cube directly from path, rather than using sample_data.py
rcm8_path = os.path.join(os.path.dirname(__file__), '..', 'deltametrics',
                         'sample_data', 'files', 'pyDeltaRCM_Output_8.nc')
rcm8cube = cube.DataCube(rcm8_path)


class TestComputeBoxyStratigraphyVolume:

    elev = rcm8cube['eta']
    time = rcm8cube['time']

    def test_returns_volume_and_elevations(self):
        s, e = strat.compute_boxy_stratigraphy_volume(self.elev,
                                                      self.time,
                                                      dz=0.05)
        assert s.ndim == 3
        assert s.shape == e.shape
        assert e[1,0,0] - e[0,0,0] == pytest.approx(0.05)

    def test_returns_volume_and_elevations_given_z(self):
        z = np.linspace(-20, 500, 200)
        s, e = strat.compute_boxy_stratigraphy_volume(self.elev,
                                                      self.time,
                                                      z=z)
        assert s.ndim == 3
        assert s.shape == e.shape
        assert np.all(e[:,0,0] == z)

    @pytest.mark.xfail(raises=NotImplementedError, strict=True, reason='Not yet developed.')
    def test_return_cube(self):
        s, e = strat.compute_boxy_stratigraphy_volume(self.elev, self.time,
                                                      dz=0.05,
                                                      return_cube=True)

    def test_lessthan3d_error(self):
        with pytest.raises(ValueError, match=r'Input arrays must be three-dimensional.'):
            strat.compute_boxy_stratigraphy_volume(self.elev[:, 10, 120].squeeze(),
                                                   self.time[:, 10, 120].squeeze(),
                                                   dz=0.05)
        with pytest.raises(ValueError, match=r'Input arrays must be three-dimensional.'):
            strat.compute_boxy_stratigraphy_volume(self.elev[:, 10, :].squeeze(),
                                                   self.time[:, 10, :].squeeze(),
                                                   dz=0.05)

    def test_bad_shape_error(self):
        with pytest.raises(ValueError, match=r'Input arrays must be three-dimensional.'):
            strat.compute_boxy_stratigraphy_volume(self.elev[:, 10, 120].squeeze(),
                                                   self.time[:, 10, 120].squeeze(),
                                                   dz=0.05)

    def test_no_z_options(self):
        with pytest.raises(ValueError, match=r'You must specify "z", "dz", or "nz.'):
            strat.compute_boxy_stratigraphy_volume(self.elev, self.time)


class TestComputeBoxyStratigraphyCoordinates:

    elev = rcm8cube['eta']
    time = rcm8cube['time']

    def test_returns_sc_dc(self):
        sc, dc = strat.compute_boxy_stratigraphy_coordinates(self.elev, dz=0.05)
        assert sc.shape == dc.shape
        assert sc.shape[1] == 3

    def test_returns_sc_dc_return_strata(self):
        sc, dc, s = strat.compute_boxy_stratigraphy_coordinates(self.elev, dz=0.05,
                                                                return_strata=True)
        assert s.ndim == 3
        assert s.shape == self.elev.shape
        assert sc.shape == dc.shape
        assert sc.shape[1] == 3

    def test_returns_sc_dc_given_z(self):
        z = np.linspace(0, 0.25, 7)
        sc, dc = strat.compute_boxy_stratigraphy_coordinates(self.elev, z=z)
        assert np.min(sc[:, 0]) == 0
        assert np.max(sc[:, 0]) == 6

    @pytest.mark.xfail(raises=NotImplementedError, strict=True, reason='Not yet developed.')
    def test_return_cube(self):
        s, sc, dc = strat.compute_boxy_stratigraphy_coordinates(self.elev,
                                                      dz=0.05, return_cube=True)

    def test_no_z_options(self):
        with pytest.raises(ValueError, match=r'You must specify "z", "dz", or "nz.'):
            strat.compute_boxy_stratigraphy_coordinates(self.elev[:, 10, 120].squeeze())


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
        s, p = strat._compute_elevation_to_preservation(np.array([0, 0, 0, 0]))
        assert np.all(s == np.array([0, 0, 0, 0]))
        assert np.all(p == np.array([False, False, False, False]))

    def test_1d_all_ones(self):
        s, p = strat._compute_elevation_to_preservation(np.array([1, 1, 1, 1]))
        assert np.all(s == np.array([1, 1, 1, 1]))
        assert np.all(p == np.array([False, False, False, False]))

    def test_1d_all_aggrade(self):
        s, p = strat._compute_elevation_to_preservation(np.array([0, 1, 2, 3]))
        assert np.all(s == np.array([0, 1, 2, 3]))
        assert np.all(p == np.array([True, True, True, True]))
        assert np.all(s[1:] - s[:-1] == 1)

    def test_1d_all_erode_positive(self):
        s, p = strat._compute_elevation_to_preservation(np.array([3, 2, 1, 0]))
        assert np.all(s == np.array([0, 0, 0, 0]))
        assert np.all(p == np.array([False, False, False, False]))

    def test_1d_all_erode_negative(self):
        s, p = strat._compute_elevation_to_preservation(np.array([0, -1, -2, -3]))
        assert np.all(s == np.array([-3, -3, -3, -3]))
        assert np.all(p == np.array([False, False, False, False]))

    def test_1d_up_down(self):
        s, p = strat._compute_elevation_to_preservation(np.array([0, 1, 2, 1]))
        assert np.all(s == np.array([0, 1, 1, 1]))
        assert np.all(p == np.array([True, True, False, False]))

    def test_1d_up_down_flat(self):
        s, p = strat._compute_elevation_to_preservation(np.array([0, 1, 2, 1, 1]))
        assert np.all(s == np.array([0, 1, 1, 1, 1]))
        assert np.all(p == np.array([True, True, False, False, False]))

    def test_1d_up_down_up(self):
        s, p = strat._compute_elevation_to_preservation(np.array([0, 1, 2, 1, 2]))
        assert np.all(s == np.array([0, 1, 1, 1, 2]))
        assert np.all(p == np.array([True, True, False, False, True]))

    def test_1d_up_down_down(self):
        s, p = strat._compute_elevation_to_preservation(np.array([0, 1, 2, 1, 0]))
        assert np.all(s == np.array([0, 0, 0, 0, 0]))
        assert np.all(p == np.array([False, False, False, False, False]))

    def test_2d_all_zeros(self):
        s, p = strat._compute_elevation_to_preservation(np.zeros((6, 4)))
        assert np.all(s == np.zeros((6, 4)))
        assert np.all(p == np.zeros((6, 4), dtype=np.bool))

    def test_2d_all_aggrade(self):
        e = np.tile(np.arange(0, 3), (2, 1)).T
        s, p = strat._compute_elevation_to_preservation(e)
        assert np.all(s == np.array([[0, 0], [1, 1], [2, 2]]))
        assert np.all(p == np.ones((3, 2), dtype=np.bool))

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
        assert np.all(p == np.zeros((6, 4, 4), dtype=np.bool))

    def test_3d_all_aggrade(self):
        e = np.tile(np.arange(0, 3), (2, 2, 1)).T
        s, p = strat._compute_elevation_to_preservation(e)
        assert np.all(s == np.array([[[0, 0], [0, 0]], [[1, 1], [1, 1]], [[2, 2], [2, 2]]]))
        assert np.all(p == np.ones((3, 2, 2), dtype=np.bool))

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
        # e = np.expand_dims(e, axis=(1,2))  # expand elevation to work with `strat` funcs
        z = strat._determine_strat_coordinates(e, dz=0.5)  # vert coordinates
        assert z[0] == 0
        assert z[-1] == 3
        assert len(z) == 7
        s, p = strat._compute_elevation_to_preservation(e)  # strat and preservation
        sc, dc = strat._compute_preservation_to_cube(s, z)
        lst = np.argmin(s[:, ...] < s[-1, ...], axis=0)  # last elevation idx
        c = self.take_var_time(s, z, sc, dc)
        assert s[-1] == e[-1]
        assert s[-1] == e[-1]
        assert np.all(np.isnan(c[2:]))  # from z>=1 to z==3
        assert np.all(c[:2] == 0)
        assert lst == 1

    def test_onedim_traj_drop_at_end_to_zero(self):
        e = np.array([0, 1, 1, 0])
        # e = np.expand_dims(e, axis=(1,2))
        z = strat._determine_strat_coordinates(e, dz=0.5)  # vert coordinates
        s, p = strat._compute_elevation_to_preservation(e)  # strat and preservation
        sc, dc = strat._compute_preservation_to_cube(s, z)
        c = self.take_var_time(s, z, sc, dc)
        assert len(z) == 3
        assert np.all(np.isnan(c[:]))
        assert np.all(p == np.array([False, False, False, False]))

    def test_onedim_traj_upsanddowns(self):
        e = np.array([0, 0, 1, 4, 6, 5, 3.5, 5, 7, 5, 6])
        # e = np.expand_dims(e, axis=(1,2))
        z = strat._determine_strat_coordinates(e, dz=0.5)  # vert coordinates
        s, p = strat._compute_elevation_to_preservation(e)  # strat and preservation
        sc, dc = strat._compute_preservation_to_cube(s, z)
        c = self.take_var_time(s, z, sc, dc)
        assert z[-1] == 7
        assert s[-1] == 6
        assert np.all(p.nonzero()[0] == (2, 3, 7, 10))
        assert c[0] == 1

    def test_onedim_traj_upsanddowns_negatives(self):
        # e = np.array([0, 0, -1, -4, -2, 3, 3.5, 3, 3, 4, 4])
        e_xr = xr.DataArray([0, 0, -1, -4, -2, 3, 3.5, 3, 3, 4, 4])
        e = e_xr.cubevar
        z = strat._determine_strat_coordinates(e, dz=0.5)  # vert coordinates
        s, p = strat._compute_elevation_to_preservation(e)  # strat and preservation
        sc, dc = strat._compute_preservation_to_cube(s, z)
        c = self.take_var_time(s, z, sc, dc)
        assert np.all(p.nonzero()[0] == (4, 5, 9))


class TestDetermineStratCoordinates:

    def test_given_none(self):
        e = np.array([0, 1, 1, 2, 1])
        with pytest.raises(ValueError, match=r'You must *.'):
            z = strat._determine_strat_coordinates(e)

    def test_given_z(self):
        e = np.array([0, 1, 1, 2, 1])
        z_in = np.arange(0, 10, step=0.2)
        z = strat._determine_strat_coordinates(e, z=z_in)
        assert np.all(z == z_in)

    def test_given_z_scalar(self):
        e = np.array([0, 1, 1, 2, 1])
        with pytest.raises(ValueError):
            z = strat._determine_strat_coordinates(e, z=0.05)

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
            z = strat._determine_strat_coordinates(e, dz=0)

    def test_given_dz_negative(self):
        e = np.array([0, 1, 1, 2, 1])
        with pytest.raises(ValueError, match=r'"dz" or "nz" cannot *.'):
            z = strat._determine_strat_coordinates(e, dz=-0.5)

    def test_given_nz(self):
        e = np.array([0, 1, 1, 2, 1])
        z = strat._determine_strat_coordinates(e, nz=50)
        assert len(z) == 50
        assert z[-1] == 2.00
        assert z[0] == 0
        assert z[1] - z[0] == pytest.approx(2 / 49)

    def test_given_nz_negative_endpoint(self):
        e = np.array([0, 1, 1, 50, -1])
        z = strat._determine_strat_coordinates(e, nz=50)
        assert len(z) == 50
        assert z[-1] == pytest.approx(50)
        assert z[0] == -1
        assert z[1] - z[0] == pytest.approx(51 / 49)

    def test_given_nz_zero(self):
        e = np.array([0, 1, 1, 2, 1])
        with pytest.raises(ValueError, match=r'"dz" or "nz" cannot *.'):
            z = strat._determine_strat_coordinates(e, nz=0)

    def test_given_nz_negative(self):
        e = np.array([0, 1, 1, 2, 1])
        with pytest.raises(ValueError, match=r'"dz" or "nz" cannot *.'):
            z = strat._determine_strat_coordinates(e, nz=-5)

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
