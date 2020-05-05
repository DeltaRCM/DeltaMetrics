import pytest

import sys
import os

import numpy as np

from deltametrics import cube

from deltametrics import plot
from deltametrics import section


rcm8_path = os.path.join(os.path.dirname(__file__), '..', 'deltametrics',
                         'sample_data', 'files', 'pyDeltaRCM_Output_8.nc')

rcm8cube = cube.DataCube(rcm8_path)




rcm8cube.stratigraphy_from('eta')
# rcm8cube.sections['trial'].variables







rcm8cube_nostrat = cube.DataCube(rcm8_path)
rcm8cube_nostrat.register_section('demo_nostrat', section.StrikeSection(y=5))


def test_nostrat_as_spacetime_is_default():
    df = rcm8cube_nostrat.sections['demo_nostrat']['velocity']
    st = rcm8cube_nostrat.sections['demo_nostrat']['velocity'].as_spacetime()
    assert np.all(df == st)


@pytest.mark.xfail(raises=AttributeError, strict=True)
def test_nostrat_nopreservationinfo():
    st = rcm8cube_nostrat.sections['demo_nostrat']['velocity'].as_spacetime(preserved=True)


@pytest.mark.xfail(raises=AttributeError, strict=True)
def test_nostrat_nostratigraphyinfo():
    st = rcm8cube_nostrat.sections['demo_nostrat']['velocity'].as_stratigraphy()
