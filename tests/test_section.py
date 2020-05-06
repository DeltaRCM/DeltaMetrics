import pytest

import sys
import os

import numpy as np

from deltametrics import cube

from deltametrics import plot
from deltametrics import section


rcm8_path = os.path.join(os.path.dirname(__file__), '..', 'deltametrics',
                         'sample_data', 'files', 'pyDeltaRCM_Output_8.nc')


def test_StrikeSection_without_cube():
    ss = section.StrikeSection(y=5)
    assert ss.y == 5
    assert ss.cube is None


def test_StrikeSection_standalone_instantiation():
    rcm8cube = cube.DataCube(rcm8_path)
    sass = section.StrikeSection(rcm8cube, y=12)
    assert sass.y == 12
    assert sass.cube == rcm8cube
    assert sass.trace.shape[1] == 2


def test_StrikeSection_register_section():
    rcm8cube = cube.DataCube(rcm8_path)
    # rcm8cube.stratigraphy_from('eta')
    rcm8cube.register_section('test', section.StrikeSection(y=5))
    assert len(rcm8cube.sections['test'].variables) > 0


test_path = np.column_stack((np.arange(10, 110, 2),
                             np.arange(50, 150, 2)))


def test_PathSection_without_cube():
    ps = section.PathSection(path=test_path)
    assert ps._path.shape[1] == 2
    assert ps.cube is None


def test_PathSection_standalone_instantiation():
    rcm8cube = cube.DataCube(rcm8_path)
    saps = section.PathSection(rcm8cube, path=test_path)
    assert saps.cube == rcm8cube
    assert saps.trace.shape == test_path.shape
    assert len(saps.variables) > 0


def test_PathSection_register_section():
    rcm8cube = cube.DataCube(rcm8_path)
    rcm8cube.stratigraphy_from('eta')
    rcm8cube.register_section('test', section.PathSection(path=test_path))
    assert len(rcm8cube.sections['test'].variables) > 0


# register a section of each type to use for tests of strat attributes and
# methods

rcm8cube = cube.DataCube(rcm8_path)
rcm8cube.stratigraphy_from('eta')
rcm8cube.register_section('test_StrikeSection', section.StrikeSection(y=5))


def test_registered_StrikeSection_attributes():
    assert np.all(rcm8cube.sections['test_StrikeSection'].trace[:, 1] == 5)
    assert rcm8cube.sections['test_StrikeSection'].s.size == 240
    assert len(rcm8cube.sections['test_StrikeSection'].variables) > 0
    assert rcm8cube.sections['test_StrikeSection'].y == 5






rcm8cube_nostrat = cube.DataCube(rcm8_path)
rcm8cube_nostrat.register_section('test_nostrat', section.StrikeSection(y=5))

def test_nostrat_as_spacetime_is_default():
    df = rcm8cube_nostrat.sections['test_nostrat']['velocity']
    st = rcm8cube_nostrat.sections['test_nostrat']['velocity'].as_spacetime()
    assert np.all(df == st)


@pytest.mark.xfail(raises=AttributeError, strict=True)
def test_nostrat_nopreservationinfo():
    st = rcm8cube_nostrat.sections['test_nostrat']['velocity'].as_spacetime(preserved=True)


@pytest.mark.xfail(raises=AttributeError, strict=True)
def test_nostrat_nostratigraphyinfo():
    st = rcm8cube_nostrat.sections['test_nostrat']['velocity'].as_stratigraphy()
