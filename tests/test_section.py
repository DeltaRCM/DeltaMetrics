import pytest

import sys
import os

import numpy as np

from deltametrics import cube

from deltametrics import plot
from deltametrics import section


rcm8_path = os.path.join(os.path.dirname(__file__), '..', 'deltametrics',
                         'sample_data', 'files', 'pyDeltaRCM_Output_8.nc')


# Test the basics of each different section type

class TestStrikeSection:
    """Test the basic of the StrikeSection."""

    def test_StrikeSection_without_cube(self):
        ss = section.StrikeSection(y=5)
        assert ss.y == 5
        assert ss.cube is None
        assert ss.s is None
        assert np.all(ss.trace == np.array([[None, None]]))
        assert ss._x is None
        assert ss._y is None
        assert ss.variables is None

    def test_StrikeSection_bad_cube(self):
        badcube = ['some', 'list']
        with pytest.raises(TypeError, match=r'Expected type is *.'):
            sass = section.StrikeSection(badcube, y=12)

    def test_StrikeSection_standalone_instantiation(self):
        rcm8cube = cube.DataCube(rcm8_path)
        sass = section.StrikeSection(rcm8cube, y=12)
        assert sass.y == 12
        assert sass.cube == rcm8cube
        assert sass.trace.shape == (240, 2)
        assert len(sass.variables) > 0

    def test_StrikeSection_register_section(self):
        rcm8cube = cube.DataCube(rcm8_path)
        rcm8cube.register_section('test', section.StrikeSection(y=5))
        assert len(rcm8cube.sections['test'].variables) > 0
        assert rcm8cube.sections['test'].cube is rcm8cube


class TestPathSection:
    """Test the basic of the PathSection."""

    test_path = np.column_stack((np.arange(10, 110, 2),
                                 np.arange(50, 150, 2)))

    def test_PathSection_without_cube(self):
        ps = section.PathSection(path=self.test_path)
        assert ps._path.shape[1] == 2
        assert ps.cube is None

    def test_PathSection_bad_cube(self):
        badcube = ['some', 'list']
        with pytest.raises(TypeError, match=r'Expected type is *.'):
            sass = section.PathSection(badcube, path=self.test_path)

    def test_PathSection_standalone_instantiation(self):
        rcm8cube = cube.DataCube(rcm8_path)
        saps = section.PathSection(rcm8cube, path=self.test_path)
        assert saps.cube == rcm8cube
        assert saps.trace.shape == self.test_path.shape
        assert len(saps.variables) > 0

    def test_PathSection_register_section(self):
        rcm8cube = cube.DataCube(rcm8_path)
        rcm8cube.stratigraphy_from('eta')
        rcm8cube.register_section('test', section.PathSection(path=self.test_path))
        assert len(rcm8cube.sections['test'].variables) > 0
        assert isinstance(rcm8cube.sections['test'], section.PathSection)


class TestCubesWithManySections:

    rcm8cube = cube.DataCube(rcm8_path)
    sc8cube = cube.StratigraphyCube.from_DataCube(rcm8cube)
    test_path = np.column_stack((np.arange(10, 110, 2),
                                 np.arange(50, 150, 2)))

    def test_data_equivalence(self):
        assert self.rcm8cube.dataio is self.sc8cube.dataio
        assert np.all(self.rcm8cube.dataio['time'] ==
                      self.sc8cube.dataio['time'])
        assert np.all(self.rcm8cube.dataio['velocity'] ==
                      self.sc8cube.dataio['velocity'])

    def test_register_multiple_strikes(self):
        self.rcm8cube.register_section('test1', section.StrikeSection(y=5))
        self.rcm8cube.register_section('test2', section.StrikeSection(y=5))
        self.rcm8cube.register_section('test3', section.StrikeSection(y=8))
        self.rcm8cube.register_section('test4', section.StrikeSection(y=10))
        assert not self.rcm8cube.sections['test1'] is self.rcm8cube.sections['test2']
        assert np.all(self.rcm8cube.sections['test1']['velocity'] ==
                      self.rcm8cube.sections['test2']['velocity'])
        assert not self.rcm8cube.sections['test1'] is self.rcm8cube.sections['test3']
        assert not self.rcm8cube.sections['test1'] is self.rcm8cube.sections['test4']
        assert not np.all(self.rcm8cube.sections['test1']['velocity'] ==
                          self.rcm8cube.sections['test3']['velocity'])

    def test_register_strike_and_path(self):
        self.rcm8cube.register_section('test1', section.StrikeSection(y=5))
        self.rcm8cube.register_section('test1a', section.StrikeSection(y=5))
        self.rcm8cube.register_section('test2', section.PathSection(path=self.test_path))
        assert not self.rcm8cube.sections['test1'] is self.rcm8cube.sections['test2']
        assert self.rcm8cube.sections['test1'].trace.shape == \
               self.rcm8cube.sections['test1a'].trace.shape
        # create alias and verify differences
        t1, t2 = self.rcm8cube.sections['test1'], self.rcm8cube.sections['test2']
        assert not t1 is t2


# test the core functionality common to all section types, for different Cubes and strat

class TestSectionFromDataCubeNoStratigraphy:
    
    rcm8cube = cube.DataCube(rcm8_path)
    rcm8cube.register_section('test', section.StrikeSection(y=5))

    def test_not_knows_stratigraphy(self):
        df = self.rcm8cube.sections['test']['velocity']._knows_stratigraphy is False
        with pytest.raises(AttributeError, match=r'No preservation information.'):
            self.rcm8cube.sections['test']['velocity'].knows_stratigraphy

    def test_nostratigraphyinfo(self):
        with pytest.raises(AttributeError, match=r'No preservation information.'):
            st = self.rcm8cube.sections['test']['velocity'].as_stratigraphy()

    def test_SectionVariable_basic_math_comparisons(self):
        s1 = self.rcm8cube.sections['test']['velocity']
        s2 = self.rcm8cube.sections['test']['depth']
        s3 = np.absolute(self.rcm8cube.sections['test']['eta'])
        assert np.all(s1 + s1 == s1 * 2)
        assert not np.any((s2 - np.random.rand(*s2.shape)) == s2)
        assert np.all(s3 + s3 > s3)
        assert type(s3) is section.DataSectionVariable

    def test_trace(self):
        assert isinstance(self.rcm8cube.sections['test'].trace, np.ndarray)

    def test_s(self):
        assert isinstance(self.rcm8cube.sections['test'].s, np.ndarray)

    def test_z(self):
        assert isinstance(self.rcm8cube.sections['test'].z, np.ndarray)

    def test_variables(self):
        assert isinstance(self.rcm8cube.sections['test'].variables, list)


class TestSectionFromDataCubeWithStratigraphy:

    rcm8cube = cube.DataCube(rcm8_path)
    rcm8cube.stratigraphy_from('eta')
    rcm8cube.register_section('test', section.StrikeSection(y=5))

    def test_registered_StrikeSection_attributes(self):
        assert np.all(self.rcm8cube.sections['test'].trace[:, 1] == 5)
        assert self.rcm8cube.sections['test'].s.size == 240
        assert len(self.rcm8cube.sections['test'].variables) > 0
        assert self.rcm8cube.sections['test'].y == 5

    def test_SectionVariable_basic_math(self):
        s1 = self.rcm8cube.sections['test']['velocity']
        assert np.all(s1 + s1 == s1 * 2)


class TestSectionFromStratigraphyCube:

    rcm8cube = cube.DataCube(rcm8_path)
    sc8cube = cube.StratigraphyCube.from_DataCube(rcm8cube)
    rcm8cube.register_section('test', section.StrikeSection(y=5))
    sc8cube.register_section('test', section.StrikeSection(y=5))

    def test_nonequal_sections(self):
        assert not self.rcm8cube.sections['test'] is self.sc8cube.sections['test']

    def test_trace(self):
        assert isinstance(self.rcm8cube.sections['test'].trace, np.ndarray)
        assert isinstance(self.sc8cube.sections['test'].trace, np.ndarray)

    def test_s(self):
        assert isinstance(self.rcm8cube.sections['test'].s, np.ndarray)
        assert isinstance(self.sc8cube.sections['test'].s, np.ndarray)

    def test_z(self):
        assert isinstance(self.rcm8cube.sections['test'].z, np.ndarray)
        assert isinstance(self.sc8cube.sections['test'].z, np.ndarray)

    def test_variables(self):
        assert isinstance(self.rcm8cube.sections['test'].variables, list)
        assert isinstance(self.sc8cube.sections['test'].variables, list)