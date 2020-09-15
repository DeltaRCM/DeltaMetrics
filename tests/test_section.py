import pytest

import sys
import os

import numpy as np
import matplotlib.pyplot as plt

from deltametrics import cube

from deltametrics import plot
from deltametrics import section
from deltametrics import utils


rcm8_path = os.path.join(os.path.dirname(__file__), '..', 'deltametrics',
                         'sample_data', 'files', 'pyDeltaRCM_Output_8.nc')


# Test the basics of each different section type

class TestStrikeSection:
    """Test the basic of the StrikeSection."""

    def test_StrikeSection_without_cube(self):
        ss = section.StrikeSection(y=5)
        assert ss.y == 5
        assert ss.shape is None
        assert ss.cube is None
        assert ss.s is None
        assert np.all(ss.trace == np.array([[None, None]]))
        assert ss._x is None
        assert ss._y is None
        assert ss.variables is None
        with pytest.raises(AttributeError, match=r'No cube connected.*.'):
            ss['velocity']

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

    def test_StrikeSection_register_section_x_limits(self):
        rcm8cube = cube.DataCube(rcm8_path)
        rcm8cube.register_section('tuple', section.StrikeSection(y=5,
                                                                x=(10, 110)))
        rcm8cube.register_section('list', section.StrikeSection(y=5,
                                                                x=(20, 110)))
        assert len(rcm8cube.sections) == 2
        assert rcm8cube.sections['tuple']._x.shape[0] == 100
        assert rcm8cube.sections['list']._x.shape[0] == 90
        assert np.all(rcm8cube.sections['list']._y == 5)
        assert np.all(rcm8cube.sections['tuple']._y == 5)


class TestPathSection:
    """Test the basic of the PathSection."""

    test_path = np.column_stack((np.arange(10, 110, 20),
                                 np.arange(50, 150, 20)))

    def test_without_cube(self):
        ps = section.PathSection(path=self.test_path)
        assert ps.path is None
        assert ps.shape is None
        assert ps.cube is None
        assert ps.s is None
        assert np.all(ps.trace == np.array([[None, None]]))
        assert ps._x is None
        assert ps._y is None
        assert ps.variables is None
        with pytest.raises(AttributeError, match=r'No cube connected.*.'):
            ps['velocity']

    def test_bad_cube(self):
        badcube = ['some', 'list']
        with pytest.raises(TypeError, match=r'Expected type is *.'):
            saps = section.PathSection(badcube, path=self.test_path)

    def test_standalone_instantiation(self):
        rcm8cube = cube.DataCube(rcm8_path)
        saps = section.PathSection(rcm8cube, path=self.test_path)
        assert saps.cube == rcm8cube
        assert saps.trace.shape[0] > 20
        assert saps.trace.shape[1] == self.test_path.shape[1]
        assert len(saps.variables) > 0

    def test_register_section(self):
        rcm8cube = cube.DataCube(rcm8_path)
        rcm8cube.stratigraphy_from('eta')
        rcm8cube.register_section(
            'test', section.PathSection(path=self.test_path))
        assert len(rcm8cube.sections['test'].variables) > 0
        assert isinstance(rcm8cube.sections['test'], section.PathSection)
        assert rcm8cube.sections['test'].shape[0] > 20

    def test_return_path(self):
        # test that returned path and trace are the same
        rcm8cube = cube.DataCube(rcm8_path)
        saps = section.PathSection(rcm8cube, path=self.test_path)
        _t = saps.trace
        _p = saps.path
        assert np.all(_t == _p)

    def test_path_reduced_unique(self):
        # test a first case with a straight line
        rcm8cube = cube.DataCube(rcm8_path)
        xy = np.column_stack((np.linspace(50, 150, num=4000, dtype=np.int),
                              np.linspace(10, 90, num=4000, dtype=np.int)))
        saps1 = section.PathSection(rcm8cube, path=xy)
        assert saps1.path.shape != xy.shape
        assert np.all(saps1.path == np.unique(xy, axis=0))

        # test a second case with small line to ensure non-unique removed
        saps2 = section.PathSection(rcm8cube, path=np.array([[50, 25],
                                                             [50, 26],
                                                             [50, 26],
                                                             [50, 27]]))
        assert saps2.path.shape == (3, 2)


class TestCircularSection:
    """Test the basic of the CircularSection."""

    def test_without_cube(self):
        cs = section.CircularSection(radius=30)
        assert cs.shape is None
        assert cs.cube is None
        assert cs.s is None
        assert np.all(cs.trace == np.array([[None, None]]))
        assert cs._x is None
        assert cs._y is None
        assert cs.variables is None
        with pytest.raises(AttributeError, match=r'No cube connected.*.'):
            cs['velocity']

    def test_bad_cube(self):
        badcube = ['some', 'list']
        with pytest.raises(TypeError, match=r'Expected type is *.'):
            sacs = section.CircularSection(badcube, radius=30)

    def test_standalone_instantiation(self):
        rcm8cube = cube.DataCube(rcm8_path)
        sacs = section.CircularSection(rcm8cube, radius=30)
        assert sacs.cube == rcm8cube
        assert sacs.trace.shape[0] == 85
        assert len(sacs.variables) > 0
        sacs2 = section.CircularSection(rcm8cube, radius=30, origin=(10, 0))
        assert sacs2.cube == rcm8cube
        assert sacs2.trace.shape[0] == 85
        assert len(sacs2.variables) > 0
        assert scas.origin == (10, 0)

    def test_register_section(self):
        rcm8cube = cube.DataCube(rcm8_path)
        rcm8cube.stratigraphy_from('eta')
        rcm8cube.register_section(
            'test', section.CircularSection(radius=30))
        assert len(rcm8cube.sections['test'].variables) > 0
        assert isinstance(rcm8cube.sections['test'], section.CircularSection)

    def test_all_idx_reduced_unique(self):
        # we try this for a bunch of different radii
        rcm8cube = cube.DataCube(rcm8_path)
        sacs1 = section.CircularSection(rcm8cube, radius=40)
        assert len(sacs1.trace) == len(np.unique(sacs1.trace, axis=0))
        sacs2 = section.CircularSection(rcm8cube, radius=2334)
        assert len(sacs2.trace) == len(np.unique(sacs2.trace, axis=0))
        sacs3 = section.CircularSection(rcm8cube, radius=167)
        assert len(sacs3.trace) == len(np.unique(sacs3.trace, axis=0))
        sacs4 = section.CircularSection(rcm8cube, radius=33)
        assert len(sacs4.trace) == len(np.unique(sacs4.trace, axis=0))


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
        assert not self.rcm8cube.sections[
            'test1'] is self.rcm8cube.sections['test2']
        assert np.all(self.rcm8cube.sections['test1']['velocity'] ==
                      self.rcm8cube.sections['test2']['velocity'])
        assert not self.rcm8cube.sections[
            'test1'] is self.rcm8cube.sections['test3']
        assert not self.rcm8cube.sections[
            'test1'] is self.rcm8cube.sections['test4']
        assert not np.all(self.rcm8cube.sections['test1']['velocity'] ==
                          self.rcm8cube.sections['test3']['velocity'])

    def test_register_strike_and_path(self):
        self.rcm8cube.register_section('test1', section.StrikeSection(y=5))
        self.rcm8cube.register_section('test1a', section.StrikeSection(y=5))
        self.rcm8cube.register_section(
            'test2', section.PathSection(path=self.test_path))
        assert not self.rcm8cube.sections[
            'test1'] is self.rcm8cube.sections['test2']
        assert self.rcm8cube.sections['test1'].trace.shape == \
            self.rcm8cube.sections['test1a'].trace.shape
        # create alias and verify differences
        t1, t2 = self.rcm8cube.sections[
            'test1'], self.rcm8cube.sections['test2']
        assert not t1 is t2


# test the core functionality common to all section types, for different
# Cubes and strat

class TestSectionFromDataCubeNoStratigraphy:

    rcm8cube_nostrat = cube.DataCube(rcm8_path)
    rcm8cube_nostrat.register_section('test', section.StrikeSection(y=5))

    def test_nostrat_getitem_explicit(self):
        s = self.rcm8cube_nostrat.sections['test'].__getitem__('velocity')
        assert isinstance(s, section.DataSectionVariable)

    def test_nostrat_getitem_implicit(self):
        s = self.rcm8cube_nostrat.sections['test']['velocity']
        assert isinstance(s, section.DataSectionVariable)

    def test_nostrat_getitem_bad_variable(self):
        with pytest.raises(AttributeError):
            self.rcm8cube_nostrat.sections['test']['badvariablename']

    def test_nostrat_getitem_broken_cube(self):
        sass = section.StrikeSection(y=5)
        with pytest.raises(AttributeError, match=r'No cube connected.*.'):
            sass['velocity']
        # make a good section, then switch to invalidcube inside section
        temp_rcm8cube_nostrat = cube.DataCube(rcm8_path)
        temp_rcm8cube_nostrat.register_section(
            'test', section.StrikeSection(y=5))
        temp_rcm8cube_nostrat.sections['test'].cube = 'badvalue!'
        with pytest.raises(TypeError):
            a = temp_rcm8cube_nostrat.sections['test'].__getitem__('velocity')
        with pytest.raises(TypeError):
            temp_rcm8cube_nostrat.sections['test']['velocity']

    def test_nostrat_not_knows_stratigraphy(self):
        assert self.rcm8cube_nostrat.sections['test'][
            'velocity']._knows_stratigraphy is False
        assert self.rcm8cube_nostrat.sections['test'][
            'velocity'].knows_stratigraphy is False

    def test_nostrat_nostratigraphyinfo(self):
        with pytest.raises(utils.NoStratigraphyError):
            st = self.rcm8cube_nostrat.sections[
                'test']['velocity'].as_stratigraphy()
        with pytest.raises(utils.NoStratigraphyError):
            st = self.rcm8cube_nostrat.sections[
                'test']['velocity'].as_preserved()

    def test_nostrat_SectionVariable_basic_math_comparisons(self):
        s1 = self.rcm8cube_nostrat.sections['test']['velocity']
        s2 = self.rcm8cube_nostrat.sections['test']['depth']
        s3 = np.absolute(self.rcm8cube_nostrat.sections['test']['eta'])
        assert np.all(s1 + s1 == s1 * 2)
        assert not np.any((s2 - np.random.rand(*s2.shape)) == s2)
        assert np.all(s3 + s3 > s3)
        assert type(s3) is section.DataSectionVariable

    def test_nostrat_trace(self):
        assert isinstance(self.rcm8cube_nostrat.sections[
                          'test'].trace, np.ndarray)

    def test_nostrat_s(self):
        _s = self.rcm8cube_nostrat.sections['test'].s
        assert isinstance(_s, np.ndarray)
        assert np.all(_s[1:] > _s[:-1])  # monotonic increase

    def test_nostrat_z(self):
        _z = self.rcm8cube_nostrat.sections['test'].z
        assert isinstance(_z, np.ndarray)
        assert np.all(_z[1:] > _z[:-1])  # monotonic increase

    def test_nostrat_variables(self):
        _v = self.rcm8cube_nostrat.sections['test'].variables
        assert len(_v) > 0
        assert isinstance(_v, list)

    def test_nostrat_show_shaded_spacetime(self):
        self.rcm8cube_nostrat.sections['test'].show('time', style='shaded',
                                                    data='spacetime')

    def test_nostrat_show_shaded_spacetime_specific_ax(self):
        fig, ax = plt.subplots()
        self.rcm8cube_nostrat.sections['test'].show('time', style='shaded',
                                                    data='spacetime', ax=ax)

    def test_nostrat_show_shaded_spacetime_no_cube(self):
        sass = section.StrikeSection(y=5)
        with pytest.raises(AttributeError, match=r'No cube connected.*.'):
            sass.show('time', style='shaded',
                      data='spacetime')

    def test_nostrat_show_shaded_aspreserved(self):
        with pytest.raises(utils.NoStratigraphyError):
            self.rcm8cube_nostrat.sections['test'].show('time', style='shaded',
                                                        data='preserved')

    def test_nostrat_show_shaded_asstratigraphy(self):
        with pytest.raises(utils.NoStratigraphyError):
            self.rcm8cube_nostrat.sections['test'].show('time', style='shaded',
                                                        data='stratigraphy')

    def test_nostrat_show_lines_spacetime(self):
        self.rcm8cube_nostrat.sections['test'].show('time', style='lines',
                                                    data='spacetime')

    def test_nostrat_show_lines_aspreserved(self):
        with pytest.raises(utils.NoStratigraphyError):
            self.rcm8cube_nostrat.sections['test'].show('time', style='lines',
                                                        data='preserved')

    def test_nostrat_show_lines_asstratigraphy(self):
        with pytest.raises(utils.NoStratigraphyError):
            self.rcm8cube_nostrat.sections['test'].show('time', style='lines',
                                                        data='stratigraphy')

    def test_nostrat_show_bad_style(self):
        with pytest.raises(ValueError, match=r'Bad style argument: "somethinginvalid"'):
            self.rcm8cube_nostrat.sections['test'].show('time', style='somethinginvalid',
                                                        data='spacetime', label=True)

    def test_nostrat_show_bad_variable(self):
        with pytest.raises(AttributeError):
            self.rcm8cube_nostrat.sections['test'].show('badvariablename')

    def test_nostrat_show_label_true(self):
        # no assertions, just functionality test
        self.rcm8cube_nostrat.sections['test'].show('time', label=True)

    def test_nostrat_show_label_given(self):
        # no assertions, just functionality test
        self.rcm8cube_nostrat.sections['test'].show('time', label='TESTLABEL!')


class TestSectionFromDataCubeWithStratigraphy:

    rcm8cube = cube.DataCube(rcm8_path)
    rcm8cube.stratigraphy_from('eta')
    rcm8cube.register_section('test', section.StrikeSection(y=5))

    def test_withstrat_getitem_explicit(self):
        s = self.rcm8cube.sections['test'].__getitem__('velocity')
        assert isinstance(s, section.DataSectionVariable)

    def test_withstrat_getitem_implicit(self):
        s = self.rcm8cube.sections['test']['velocity']
        assert isinstance(s, section.DataSectionVariable)

    def test_withstrat_getitem_bad_variable(self):
        with pytest.raises(AttributeError):
            self.rcm8cube.sections['test']['badvariablename']

    def test_withstrat_getitem_broken_cube(self):
        sass = section.StrikeSection(y=5)
        with pytest.raises(AttributeError, match=r'No cube connected.*.'):
            sass['velocity']
        # make a good section, then switch to invalidcube inside section
        temp_rcm8cube = cube.DataCube(rcm8_path)
        temp_rcm8cube.register_section('test', section.StrikeSection(y=5))
        temp_rcm8cube.sections['test'].cube = 'badvalue!'
        with pytest.raises(TypeError):
            a = temp_rcm8cube.sections['test'].__getitem__('velocity')
        with pytest.raises(TypeError):
            temp_rcm8cube.sections['test']['velocity']

    def test_withstrat_knows_stratigraphy(self):
        assert self.rcm8cube.sections['test'][
            'velocity']._knows_stratigraphy is True
        assert self.rcm8cube.sections['test'][
            'velocity'].knows_stratigraphy is True

    def test_withstrat_trace(self):
        assert isinstance(self.rcm8cube.sections['test'].trace, np.ndarray)

    def test_withstrat_s(self):
        _s = self.rcm8cube.sections['test'].s
        assert isinstance(_s, np.ndarray)
        assert np.all(_s[1:] > _s[:-1])  # monotonic increase

    def test_withstrat_z(self):
        _z = self.rcm8cube.sections['test'].z
        assert isinstance(_z, np.ndarray)
        assert np.all(_z[1:] > _z[:-1])  # monotonic increase

    def test_withstrat_variables(self):
        _v = self.rcm8cube.sections['test'].variables
        assert len(_v) > 0
        assert isinstance(_v, list)

    def test_withstrat_registered_StrikeSection_attributes(self):
        assert np.all(self.rcm8cube.sections['test'].trace[:, 1] == 5)
        assert self.rcm8cube.sections['test'].s.size == 240
        assert len(self.rcm8cube.sections['test'].variables) > 0
        assert self.rcm8cube.sections['test'].y == 5

    def test_withstrat_SectionVariable_basic_math(self):
        s1 = self.rcm8cube.sections['test']['velocity']
        assert np.all(s1 + s1 == s1 * 2)

    def test_withstrat_strat_attr_mesh_components(self):
        sa = self.rcm8cube.sections['test']['velocity'].strat_attr
        assert 'strata' in sa.keys()
        assert 'psvd_idx' in sa.keys()
        assert 'psvd_flld' in sa.keys()
        assert 'x0' in sa.keys()
        assert 'x1' in sa.keys()
        assert 's' in sa.keys()
        assert 's_sp' in sa.keys()
        assert 'z_sp' in sa.keys()

    def test_withstrat_strat_attr_shapes(self):
        sa = self.rcm8cube.sections['test']['velocity'].strat_attr
        assert sa['x0'].shape == (51, 240)
        assert sa['x1'].shape == (51, 240)
        assert sa['s'].shape == (240,)
        assert sa['s_sp'].shape == sa['z_sp'].shape

    def test_withstrat_show_shaded_spacetime(self):
        self.rcm8cube.sections['test'].show('time', style='shaded',
                                            data='spacetime')

    def test_withstrat_show_shaded_spacetime_specific_ax(self):
        fig, ax = plt.subplots()
        self.rcm8cube.sections['test'].show('time', style='shaded',
                                            data='spacetime', ax=ax)

    def test_withstrat_show_shaded_spacetime_no_cube(self):
        sass = section.StrikeSection(y=5)
        with pytest.raises(AttributeError, match=r'No cube connected.*.'):
            sass.show('time', style='shaded',
                      data='spacetime')

    def test_withstrat_show_shaded_aspreserved(self):
        self.rcm8cube.sections['test'].show('time', style='shaded',
                                            data='preserved')

    def test_withstrat_show_shaded_asstratigraphy(self):
        self.rcm8cube.sections['test'].show('time', style='shaded',
                                            data='stratigraphy')

    def test_withstrat_show_lines_spacetime(self):
        self.rcm8cube.sections['test'].show('time', style='lines',
                                            data='spacetime')

    def test_withstrat_show_lines_aspreserved(self):
        self.rcm8cube.sections['test'].show('time', style='lines',
                                            data='preserved')

    def test_withstrat_show_lines_asstratigraphy(self):
        self.rcm8cube.sections['test'].show('time', style='lines',
                                            data='stratigraphy')

    def test_withstrat_show_bad_style(self):
        with pytest.raises(ValueError, match=r'Bad style argument: "somethinginvalid"'):
            self.rcm8cube.sections['test'].show('time', style='somethinginvalid',
                                                data='spacetime', label=True)

    def test_withstrat_show_bad_variable(self):
        with pytest.raises(AttributeError):
            self.rcm8cube.sections['test'].show('badvariablename')

    def test_withstrat_show_label_true(self):
        # no assertions, just functionality test
        self.rcm8cube.sections['test'].show('time', label=True)

    def test_withstrat_show_label_given(self):
        # no assertions, just functionality test
        self.rcm8cube.sections['test'].show('time', label='TESTLABEL!')


class TestSectionFromStratigraphyCube:

    rcm8cube = cube.DataCube(rcm8_path)
    sc8cube = cube.StratigraphyCube.from_DataCube(rcm8cube)
    rcm8cube.register_section('test', section.StrikeSection(y=5))
    sc8cube.register_section('test', section.StrikeSection(y=5))

    def test_strat_getitem_explicit(self):
        s = self.sc8cube.sections['test'].__getitem__('velocity')
        assert isinstance(s, section.StratigraphySectionVariable)

    def test_strat_getitem_implicit(self):
        s = self.sc8cube.sections['test']['velocity']
        assert isinstance(s, section.StratigraphySectionVariable)

    def test_strat_getitem_bad_variable(self):
        with pytest.raises(AttributeError):
            self.sc8cube.sections['test']['badvariablename']

    def test_strat_getitem_broken_cube(self):
        sass = section.StrikeSection(y=5)
        with pytest.raises(AttributeError, match=r'No cube connected.*.'):
            sass['velocity']
        # make a good section, then switch to invalidcube inside section
        temp_rcm8cube = cube.DataCube(rcm8_path)
        temp_rcm8cube.register_section('test', section.StrikeSection(y=5))
        temp_rcm8cube.sections['test'].cube = 'badvalue!'
        with pytest.raises(TypeError):
            a = temp_rcm8cube.sections['test'].__getitem__('velocity')
        with pytest.raises(TypeError):
            temp_rcm8cube.sections['test']['velocity']

    def test_nonequal_sections(self):
        assert not self.rcm8cube.sections[
            'test'] is self.sc8cube.sections['test']

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

    def test_strat_show_noargs(self):
        self.sc8cube.sections['test'].show('time')

    def test_strat_show_shaded_spacetime(self):
        with pytest.raises(AttributeError, match=r'No "spacetime" or "preserved"*.'):
            self.sc8cube.sections['test'].show('time', style='shaded',
                                               data='spacetime')

    def test_strat_show_shaded_spacetime_no_cube(self):
        sass = section.StrikeSection(y=5)
        with pytest.raises(AttributeError, match=r'No cube connected.*.'):
            sass.show('time', style='shaded',
                      data='spacetime')

    def test_strat_show_shaded_aspreserved(self):
        with pytest.raises(AttributeError, match=r'No "spacetime" or "preserved"*.'):
            self.sc8cube.sections['test'].show('time', style='shaded',
                                               data='preserved')

    def test_strat_show_shaded_asstratigraphy(self):
        self.sc8cube.sections['test'].show('time', style='shaded',
                                           data='stratigraphy')

    def test_strat_show_shaded_asstratigraphy_specific_ax(self):
        fig, ax = plt.subplots()
        self.sc8cube.sections['test'].show('time', style='shaded',
                                           data='stratigraphy', ax=ax)

    def test_strat_show_lines_spacetime(self):
        with pytest.raises(AttributeError, match=r'No "spacetime" or "preserved"*.'):
            self.sc8cube.sections['test'].show('time', style='lines',
                                               data='spacetime')

    def test_strat_show_lines_aspreserved(self):
        with pytest.raises(AttributeError, match=r'No "spacetime" or "preserved"*.'):
            self.sc8cube.sections['test'].show('time', style='lines',
                                               data='preserved')

    @pytest.mark.xfail(reason='not yet decided best way to implement')
    def test_strat_show_lines_asstratigraphy(self):
        self.sc8cube.sections['test'].show('time', style='lines',
                                           data='stratigraphy')

    def test_strat_show_bad_style(self):
        with pytest.raises(ValueError, match=r'Bad style argument: "somethinginvalid"'):
            self.sc8cube.sections['test'].show('time', style='somethinginvalid',
                                               data='spacetime', label=True)

    def test_strat_show_bad_variable(self):
        with pytest.raises(AttributeError):
            self.sc8cube.sections['test'].show('badvariablename')

    def test_strat_show_label_true(self):
        # no assertions, just functionality test
        self.sc8cube.sections['test'].show('time', label=True)

    def test_strat_show_label_given(self):
        # no assertions, just functionality test
        self.sc8cube.sections['test'].show('time', label='TESTLABEL!')


class TestDataSectionVariableNoStratigraphy:

    rcm8cube = cube.DataCube(rcm8_path)
    rcm8cube.register_section('test', section.StrikeSection(y=5))
    dsv = rcm8cube.sections['test']['velocity']

    def test_dsv_view_from(self):
        _arr = self.dsv + 5  # takes a view from
        assert not _arr is self.dsv
        _arr2 = (_arr - 5)
        assert np.all(_arr2 == pytest.approx(self.dsv, abs=1e-6))

    def test_dsv_instantiate_directly(self):
        _arr = np.random.rand(100, 200)
        _s = np.arange(200)
        _z = np.linspace(0, 10, num=100)
        _dsv = section.DataSectionVariable(_arr, _s, _z)
        assert isinstance(_dsv, section.DataSectionVariable)
        assert np.all(_dsv == _arr)
        assert np.all(_dsv._s == _s)
        assert np.all(_dsv._z == _z)
        assert _dsv.shape == (100, 200)
        assert _dsv._Z.shape == (100, 200)
        assert _dsv._S.shape == (100, 200)
        assert _dsv._psvd_mask is None

    def test_dsv_instantiate_directly_bad_coords_shapes(self):
        _arr = np.random.rand(100, 200)
        _s = np.arange(200)
        _z = np.linspace(0, 10, num=50)
        with pytest.raises(ValueError):
            _dsv = section.DataSectionVariable(_arr, _s, _z)
        
    def test_dsv_knows_stratigraphy(self):
        assert self.dsv._knows_stratigraphy is False
        assert self.dsv.knows_stratigraphy is False
        assert self.dsv.knows_stratigraphy == self.dsv._knows_stratigraphy

    def test_dsv__check_knows_stratigraphy(self):
        with pytest.raises(utils.NoStratigraphyError):
            self.dsv._check_knows_stratigraphy()

    def test_dsv_as_preserved(self):
        with pytest.raises(utils.NoStratigraphyError):
            self.dsv.as_preserved()

    def test_dsv_as_stratigraphy(self):
        with pytest.raises(utils.NoStratigraphyError):
            self.dsv.as_stratigraphy()


class TestDataSectionVariableWithStratigraphy:

    rcm8cube = cube.DataCube(rcm8_path)
    rcm8cube.stratigraphy_from('eta')
    rcm8cube.register_section('test', section.StrikeSection(y=5))
    dsv = rcm8cube.sections['test']['velocity']

    def test_dsv_instantiate_directly(self):
        _arr = np.random.rand(100, 200)
        _s = np.arange(200)
        _z = np.linspace(0, 10, num=100)
        _mask = np.random.randint(0, 2, (100, 200), dtype=np.bool)
        _dsv = section.DataSectionVariable(_arr, _s, _z)
        assert isinstance(_dsv, section.DataSectionVariable)
        assert np.all(_dsv == _arr)
        assert np.all(_dsv._s == _s)
        assert np.all(_dsv._z == _z)
        assert _dsv.shape == (100, 200)
        assert _dsv._Z.shape == (100, 200)
        assert _dsv._S.shape == (100, 200)
        assert _dsv._psvd_mask is None

    def test_dsv_instantiate_directly_bad_shape_mask(self):
        _arr = np.random.rand(100, 200)
        _s = np.arange(200)
        _z = np.linspace(0, 10, num=100)
        _mask = np.random.randint(0, 2, (20, 200), dtype=np.bool)
        with pytest.raises(ValueError, match=r'Shape of "_psvd_mask"*.'):
            _dsv = section.DataSectionVariable(_arr, _s, _z, _mask)

    def test_dsv_instantiate_directly_bad_shape_coord(self):
        _arr = np.random.rand(100, 200)
        _s = np.arange(500)
        _z = np.linspace(0, 10, num=100)
        _mask = np.random.randint(0, 2, (100, 200), dtype=np.bool)
        with pytest.raises(ValueError, match=r'Shape of "_s"*.'):
            _dsv = section.DataSectionVariable(_arr, _s, _z, _mask)

    def test_dsv_knows_stratigraphy(self):
        assert self.dsv._knows_stratigraphy is True
        assert self.dsv.knows_stratigraphy is True
        assert self.dsv.knows_stratigraphy == self.dsv._knows_stratigraphy

    def test_dsv__check_knows_stratigraphy(self):
        assert self.dsv._check_knows_stratigraphy()

    def test_dsv_as_preserved(self):
        _arr = self.dsv.as_preserved()
        assert _arr.shape == self.dsv.shape
        assert isinstance(_arr, np.ma.MaskedArray)

    def test_dsv_as_stratigraphy(self):
        _arr = self.dsv.as_stratigraphy()
        assert _arr.shape == (np.max(self.dsv.strat_attr['z_sp']) + 1,
                              self.dsv.shape[1])


class TestStratigraphySectionVariable:

    rcm8cube = cube.DataCube(rcm8_path)
    sc8cube = cube.StratigraphyCube.from_DataCube(rcm8cube)
    sc8cube.register_section('test', section.StrikeSection(y=5))
    ssv = sc8cube.sections['test']['velocity']

    def test_ssv_view_from(self):
        _arr = self.ssv + 5  # takes a view from
        assert not _arr is self.ssv
        assert np.all(np.isnan(_arr) == np.isnan(self.ssv))
        _arr2 = (_arr - 5)
        assert np.all(_arr2[~np.isnan(_arr2)].flatten() == pytest.approx(self.ssv[~np.isnan(self.ssv)].flatten()))

    def test_ssv_instantiate_directly_from_array(self):
        _arr = np.random.rand(100, 200)
        _s = np.arange(200)
        _z = np.linspace(0, 10, num=100)
        _ssv = section.StratigraphySectionVariable(_arr, _s, _z)
        assert isinstance(_ssv, section.StratigraphySectionVariable)
        assert np.all(_ssv == _arr)
        assert np.all(_ssv._s == _s)
        assert np.all(_ssv._z == _z)
        assert _ssv.shape == (100, 200)
        assert _ssv._Z.shape == (100, 200)
        assert _ssv._S.shape == (100, 200)
        assert _ssv._psvd_mask is None

    def test_ssv_instantiate_directly_bad_addtl_argument(self):
        _arr = np.random.rand(100, 200)
        _s = np.arange(200)
        _z = np.linspace(0, 10, num=100)
        _mask = np.copy(_arr)
        with pytest.raises(TypeError):
            _ssv = section.StratigraphySectionVariable(_arr, _s, _z, _mask)

    def test_ssv_instantiate_directly_bad_shape_coord(self):
        _arr = np.random.rand(100, 200)
        _s = np.arange(200)
        _z = np.linspace(0, 10, num=50)
        with pytest.raises(ValueError, match=r'Shape of "_s"*.'):
            _ssv = section.StratigraphySectionVariable(_arr, _s, _z)

    def test_ssv_knows_spacetime(self):
        assert self.ssv._knows_spacetime is False
        assert self.ssv.knows_spacetime is False
        assert self.ssv.knows_spacetime == self.ssv._knows_spacetime

    def test_ssv__check_knows_spacetime(self):
        with pytest.raises(AttributeError, match=r'No "spacetime" or "preserved"*.'):
            self.ssv._check_knows_spacetime()
