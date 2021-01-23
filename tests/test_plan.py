import pytest

import sys
import os

import numpy as np

from deltametrics.sample_data.cube import rcm8

from deltametrics import mask
from deltametrics import plan


class TestShorelineRoughness:

    rcm8 = rcm8()
    lm = mask.LandMask(rcm8['eta'][-1, :, :])
    sm = mask.ShorelineMask(rcm8['eta'][-1, :, :])

    def test_compute_shoreline_roughness_rcm8(self):
        # test it with default options
        rgh_0 = plan.compute_shoreline_roughness(self.sm, self.lm)
        assert rgh_0 == pytest.approx(5.5882170024)

        # test that it ignores return_line arg
        rgh_1 = plan.compute_shoreline_roughness(self.sm, self.lm,
                                                 return_line=False)
        assert rgh_1 == rgh_0

        # test that it is the same with opposite side origin
        rgh_2 = plan.compute_shoreline_roughness(self.sm, self.lm,
                                                 origin=[0, self.rcm8.shape[1]])
        assert rgh_2 == rgh_0

    def test_compute_shoreline_roughness_asarray(self):
        pass


class TestShorelineLength:

    def test_compute_shoreline_length(self):
        pass
