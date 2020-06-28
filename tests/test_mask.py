"""Tests for the mask.py script."""
import pytest
import sys
import os
import numpy as np

from deltametrics import cube
from deltametrics import mask

# initialize a cube directly from path, rather than using sample_data.py
rcm8_path = os.path.join(os.path.dirname(__file__), '..', 'deltametrics',
                         'sample_data', 'files', 'pyDeltaRCM_Output_8.nc')
rcm8cube = cube.DataCube(rcm8_path)


class TestShoreMask:
    """Tests associated with the mask.ShoreMask class."""

    def test_default_vals(self):
        """Test that default values are assigned."""
        # define the mask
        shoremask = mask.ShoreMask(rcm8cube['eta'][-1, :, :])
        # make assertions
        assert shoremask.topo_threshold == -0.5
        assert shoremask.angle_threshold == 75
        assert shoremask.numviews == 3
        assert shoremask.is_mask is False
        assert shoremask.mask_type == 'shore'

    def test_maskError(self):
        """Test that ValueError is raised if is_mask is invalid."""
        with pytest.raises(ValueError):
            shoremask = mask.ShoreMask(rcm8cube['eta'][-1, :, :],
                                       is_mask='invalid')

    def test_maskTrue(self):
        """Test that is_mask is True works."""
        # define the mask
        shoremask = mask.ShoreMask(rcm8cube['eta'][-1, :, :],
                                   is_mask=True)
        # do assertion
        assert np.all(shoremask.mask == rcm8cube['eta'][-1, :, :])

    def test_assign_vals(self):
        """Test that specified values are assigned."""
        # define the mask
        shoremask = mask.ShoreMask(rcm8cube['eta'][-1, :, :],
                                   topo_threshold=-1.0,
                                   angle_threshold=100,
                                   numviews=5)
        # make assertions
        assert shoremask.topo_threshold == -1.0
        assert shoremask.angle_threshold == 100
        assert shoremask.numviews == 5
        assert shoremask.is_mask is False

    def test_shoreline(self):
        """Check for important variables and the final mask."""
        # define the mask
        shoremask = mask.ShoreMask(rcm8cube['eta'][-1, :, :])
        # make assertions
        assert np.array_equal(shoremask.mask,
                              shoremask.mask.astype(bool)) is True
        assert hasattr(shoremask, 'oceanmap') is True
        assert hasattr(shoremask, 'mask') is True
        assert hasattr(shoremask, 'shore_image') is True


class TestLandMask:
    """Tests associated with the mask.LandMask class."""

    def test_default_vals(self):
        """Test that default values are assigned."""
        # define the mask
        landmask = mask.LandMask(rcm8cube['eta'][-1, :, :])
        # make assertions
        assert landmask.topo_threshold == -0.5
        assert landmask.angle_threshold == 75
        assert landmask.numviews == 3
        assert landmask.is_mask is False
        assert landmask.mask_type == 'land'

    def test_maskError(self):
        """Test that ValueError is raised if is_mask is invalid."""
        with pytest.raises(ValueError):
            landmask = mask.LandMask(rcm8cube['eta'][-1, :, :],
                                     is_mask='invalid')

    def test_maskTrue(self):
        """Test that is_mask is True works."""
        # define the mask
        landmask = mask.LandMask(rcm8cube['eta'][-1, :, :],
                                 is_mask=True)
        # do assertion
        assert np.all(landmask.mask == rcm8cube['eta'][-1, :, :])

    def test_assign_vals(self):
        """Test that specified values are assigned."""
        # define the mask
        landmask = mask.LandMask(rcm8cube['eta'][-1, :, :],
                                 topo_threshold=-1.0,
                                 angle_threshold=100,
                                 numviews=5)
        # make assertions
        assert landmask.topo_threshold == -1.0
        assert landmask.angle_threshold == 100
        assert landmask.numviews == 5
        assert landmask.is_mask is False

    def test_land(self):
        """Check for important variables and the final mask."""
        # define the mask
        landmask = mask.LandMask(rcm8cube['eta'][-1, :, :])
        # make assertions
        assert np.array_equal(landmask.mask,
                              landmask.mask.astype(bool)) is True
        assert hasattr(landmask, 'oceanmap') is True
        assert hasattr(landmask, 'mask') is True
        assert hasattr(landmask, 'shore_image') is True

    def test_givenshore(self):
        """Test that a ShoreMask can be passed into it."""
        shoremask = mask.ShoreMask(rcm8cube['eta'][-1, :, :])
        landmask = mask.LandMask(rcm8cube['eta'][-1, :, :],
                                 shoremask=shoremask)
        # make assertions
        assert hasattr(landmask, 'shoremask') is True
        assert hasattr(landmask, 'angle_threshold') is True
        assert hasattr(landmask, 'mask') is True
        assert hasattr(landmask, 'shore_image') is True
        assert np.array_equal(landmask.mask,
                              landmask.mask.astype(bool)) is True


class TestWetMask:
    """Tests associated with the mask.WetMask class."""

    def test_default_vals(self):
        """Test that default values are assigned."""
        # define the mask
        wetmask = mask.WetMask(rcm8cube['eta'][-1, :, :])
        # make assertions
        assert wetmask.topo_threshold == -0.5
        assert wetmask.angle_threshold == 75
        assert wetmask.numviews == 3
        assert wetmask.is_mask is False
        assert wetmask.mask_type == 'wet'

    def test_maskError(self):
        """Test that ValueError is raised if is_mask is invalid."""
        with pytest.raises(ValueError):
            wetmask = mask.WetMask(rcm8cube['eta'][-1, :, :],
                                   is_mask='invalid')

    def test_maskTrue(self):
        """Test that is_mask is True works."""
        # define the mask
        wetmask = mask.WetMask(rcm8cube['eta'][-1, :, :],
                               is_mask=True)
        # do assertion
        assert np.all(wetmask.mask == rcm8cube['eta'][-1, :, :])

    def test_assign_vals(self):
        """Test that specified values are assigned."""
        # define the mask
        wetmask = mask.WetMask(rcm8cube['eta'][-1, :, :],
                               topo_threshold=-1.0,
                               angle_threshold=100,
                               numviews=5)
        # make assertions
        assert wetmask.topo_threshold == -1.0
        assert wetmask.angle_threshold == 100
        assert wetmask.numviews == 5
        assert wetmask.is_mask is False

    def test_land(self):
        """Check for important variables and the final mask."""
        # define the mask
        wetmask = mask.WetMask(rcm8cube['eta'][-1, :, :])
        # make assertions
        assert np.array_equal(wetmask.mask,
                              wetmask.mask.astype(bool)) is True
        assert hasattr(wetmask, 'oceanmap') is True
        assert hasattr(wetmask, 'mask') is True
        assert hasattr(wetmask, 'landmask') is True

    def test_givenshore(self):
        """Test that a LandMask can be passed into it."""
        landmask = mask.LandMask(rcm8cube['eta'][-1, :, :])
        wetmask = mask.WetMask(rcm8cube['eta'][-1, :, :],
                               landmask=landmask)
        # make assertions
        assert hasattr(wetmask, 'landmask') is True
        assert hasattr(wetmask, 'oceanmap') is True
        assert hasattr(wetmask, 'mask') is True
        assert hasattr(wetmask, 'landmask') is True
        assert np.array_equal(wetmask.mask,
                              wetmask.mask.astype(bool)) is True
