"""Tests for the mask.py script."""
import pytest
import sys
import os
import numpy as np

from deltametrics import cube
from deltametrics import strat

# initialize a cube directly from path, rather than using sample_data.py
rcm8_path = os.path.join(os.path.dirname(__file__), '..', 'deltametrics',
                         'sample_data', 'files', 'pyDeltaRCM_Output_8.nc')
rcm8cube = cube.DataCube(rcm8_path)


class TestOneDimStratigraphyExamples:
    """Tests for various cases of 1d stratigraphy."""

    def test_onedim_traj_drop_at_end(self):
        e = np.array([0, 0, 1, 0])
        e = np.expand_dims(e, axis=(1,2))
        s, p = strat._compute_elevation_to_preservation(e)  # strat and preservation

    def test_onedim_traj_drop_at_end(self):
        e = np.array([0, 1, 2, 3, 1])
        e = np.expand_dims(e, axis=(1,2))  # expand elevation to work with `strat` funcs
        z = strat._determine_strat_coordinates(e, dz=0.5)  # vert coordinates
        assert z[0] == 0
        assert z[-1] == 3
        assert len(z) == 7
        s, p = strat._compute_elevation_to_preservation(e)  # strat and preservation
        t = np.arange(e.shape[0])  # x-axis time array
        t3 = np.expand_dims(t, axis=(1,2)) # 3d time, for slicing
        sc, dc = strat._compute_preservation_to_cube(s, z)
        lst = np.argmin(s[:, ...] < s[-1, :, :], axis=0) # last elevation
        c = np.full((z.shape[0], 1, 1), np.nan)
        c[sc[0,:], sc[1,:], sc[2,:]] = t3[dc[0,:], dc[1,:], dc[2,:]]

        assert s[-1] == e[-1]
        assert s[-1] == e[-1]
        assert np.all(np.isnan(c[2:, 0, 0]))  # from z>=1 to z==3
        assert np.all(c[:2, 0, 0] == 0)
        print(z)
        print(c)
        raise ValueError
    

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
