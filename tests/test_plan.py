import pytest

import sys
import os

import numpy as np

from deltametrics.sample_data import _get_rcm8_path, _get_golf_path

from deltametrics import mask
from deltametrics import cube
from deltametrics import plan

simple_land = np.zeros((10, 10))
simple_shore = np.zeros((10, 10))
simple_land[:4, :] = 1
simple_land[4, 2:7] = 1
simple_shore_array = np.array([[3, 3, 4, 4, 4, 4, 4, 3, 3, 3],
                               [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]]).T
simple_shore[simple_shore_array[:, 0], simple_shore_array[:, 1]] = 1


class TestShorelineRoughness:

    rcm8_path = _get_rcm8_path()
    with pytest.warns(UserWarning):
        rcm8 = cube.DataCube(rcm8_path)

    lm = mask.LandMask(
        rcm8['eta'][-1, :, :],
        elevation_threshold=0)
    sm = mask.ShorelineMask(
        rcm8['eta'][-1, :, :],
        elevation_threshold=0)
    lm0 = mask.LandMask(
        rcm8['eta'][0, :, :],
        elevation_threshold=0)
    sm0 = mask.ShorelineMask(
        rcm8['eta'][0, :, :],
        elevation_threshold=0)

    rcm8_expected = 5.9941268725889385

    def test_simple_case(self):
        simple_rgh = plan.compute_shoreline_roughness(simple_shore, simple_land)
        exp_area = 45
        exp_len = (7*1)+(2*1.41421356)
        exp_rgh = exp_len / np.sqrt(exp_area)
        assert simple_rgh == pytest.approx(exp_rgh)

    def test_rcm8_defaults(self):
        # test it with default options
        rgh_0 = plan.compute_shoreline_roughness(self.sm, self.lm)
        assert rgh_0 == pytest.approx(self.rcm8_expected)

    def test_rcm8_ignore_return_line(self):
        # test that it ignores return_line arg
        rgh_1 = plan.compute_shoreline_roughness(self.sm, self.lm,
                                                 return_line=False)
        assert rgh_1 == pytest.approx(self.rcm8_expected)

    def test_rcm8_defaults_opposite(self):
        # test that it is the same with opposite side origin
        rgh_2 = plan.compute_shoreline_roughness(self.sm, self.lm,
                                                 origin=[0, self.rcm8.shape[1]])
        assert rgh_2 == pytest.approx(self.rcm8_expected, abs=0.1)

    def test_rcm8_fail_no_shoreline(self):
        # check raises error 
        with pytest.raises(ValueError, match=r'No pixels in shoreline mask.'):
            plan.compute_shoreline_roughness(self.sm0, self.lm)

    def test_rcm8_fail_no_land(self):
        # check raises error 
        with pytest.raises(ValueError, match=r'No pixels in land mask.'):
            plan.compute_shoreline_roughness(self.sm, self.lm0)

    def test_compute_shoreline_roughness_asarray(self):
        # test it with default options
        _smarr = np.copy(self.sm.mask)
        _lmarr = np.copy(self.lm.mask)
        assert isinstance(_smarr, np.ndarray)
        assert isinstance(_lmarr, np.ndarray)
        rgh_3 = plan.compute_shoreline_roughness(_smarr, _lmarr)
        assert rgh_3 == pytest.approx(self.rcm8_expected)


class TestShorelineLength:

    rcm8_path = _get_rcm8_path()
    with pytest.warns(UserWarning):
        rcm8 = cube.DataCube(rcm8_path)

    sm = mask.ShorelineMask(
        rcm8['eta'][-1, :, :],
        elevation_threshold=0)
    sm0 = mask.ShorelineMask(
        rcm8['eta'][0, :, :],
        elevation_threshold=0)

    rcm8_expected = 451.71224690336953

    def test_simple_case(self):
        simple_len = plan.compute_shoreline_length(
            simple_shore)
        exp_len = (7*1)+(2*1.41421356)
        assert simple_len == pytest.approx(exp_len)

    def test_simple_case_opposite(self):
        simple_len = plan.compute_shoreline_length(
            simple_shore, origin=[10, 0])
        exp_len = (7*1)+(2*1.41421356)
        assert simple_len == pytest.approx(exp_len)

    def test_simple_case_return_line(self):
        simple_len, simple_line = plan.compute_shoreline_length(
            simple_shore, return_line=True)
        exp_len = (7*1)+(2*1.41421356)
        assert simple_len == pytest.approx(exp_len)
        assert np.all(simple_line == np.fliplr(simple_shore_array))

    def test_rcm8_defaults(self):
        # test that it is the same with opposite side origin
        len_0 = plan.compute_shoreline_length(
            self.sm)
        assert len_0 == pytest.approx(self.rcm8_expected)

    def test_rcm8_defaults_opposite(self):
        # test that it is the same with opposite side origin
        len_0, line_0 = plan.compute_shoreline_length(
            self.sm, return_line=True)
        _o = [self.rcm8.shape[2], 0]
        len_1, line_1 = plan.compute_shoreline_length(
            self.sm, origin=_o, return_line=True)
        if False:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(1, 2)
            ax[0].imshow(self.sm.mask.squeeze())
            ax[1].imshow(self.sm.mask.squeeze())
            ax[0].plot(0, 0, 'ro')
            ax[1].plot(_o[0], _o[1], 'bo')
            ax[0].plot(line_0[:, 0], line_0[:, 1], 'r-')
            ax[1].plot(line_1[:, 0], line_1[:, 1], 'b-')
            plt.show(block=False)

            fig, ax = plt.subplots()
            ax.plot(np.cumsum(np.sqrt((line_0[1:, 0]-line_0[:-1, 0])**2 +
                                      (line_0[1:, 1]-line_0[:-1, 1])**2)))
            ax.plot(np.cumsum(np.sqrt((line_1[1:, 0]-line_1[:-1, 0])**2 +
                                      (line_1[1:, 1]-line_1[:-1, 1])**2)))
            plt.show()
            breakpoint()
        assert len_1 == pytest.approx(self.rcm8_expected, abs=5.0)
