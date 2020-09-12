import pytest

import sys
import os

import numpy as np
import xarray as xr

from deltametrics import cube

from deltametrics import plot
from deltametrics import section
from deltametrics import utils

# initialize a cube directly from path, rather than using sample_data.py
rcm8_path = os.path.join(os.path.dirname(__file__), '..', 'deltametrics',
                         'sample_data', 'files', 'pyDeltaRCM_Output_8.nc')


class TestDataCubeNoStratigraphy:

    # create a fixed cube for variable existing, type checks
    fixeddatacube = cube.DataCube(rcm8_path)

    def test_init_cube_from_path_rcm8(self):
        rcm8cube = cube.DataCube(rcm8_path)
        assert rcm8cube._data_path == rcm8_path
        assert rcm8cube.dataio.type == 'netcdf'
        assert rcm8cube._plan_set == {}
        assert rcm8cube._section_set == {}
        assert type(rcm8cube.varset) is plot.VariableSet

    def test_error_init_empty_cube(self):
        with pytest.raises(TypeError):
            nocube = cube.DataCube()

    def test_error_init_bad_path(self):
        with pytest.raises(FileNotFoundError):
            nocube = cube.DataCube('./nonexistent/path.nc')

    def test_error_init_bad_extension(self):
        with pytest.raises(ValueError):
            nocube = cube.DataCube('./nonexistent/path.doc')

    def test_stratigraphy_from_eta(self):
        rcm8cube = cube.DataCube(rcm8_path)
        rcm8cube.stratigraphy_from('eta')
        assert rcm8cube._knows_stratigraphy is True

    def test_init_cube_stratigraphy_argument(self):
        rcm8cube = cube.DataCube(rcm8_path, stratigraphy_from='eta')
        assert rcm8cube._knows_stratigraphy is True

    def test_stratigraphy_from_default_noargument(self):
        rcm8cube = cube.DataCube(rcm8_path)
        rcm8cube.stratigraphy_from()
        assert rcm8cube._knows_stratigraphy is True

    def test_init_with_shared_varset_prior(self):
        shared_varset = plot.VariableSet()
        rcm8cube1 = cube.DataCube(rcm8_path, varset=shared_varset)
        rcm8cube2 = cube.DataCube(rcm8_path, varset=shared_varset)
        assert type(rcm8cube1.varset) is plot.VariableSet
        assert type(rcm8cube2.varset) is plot.VariableSet
        assert rcm8cube1.varset is shared_varset
        assert rcm8cube1.varset is rcm8cube2.varset

    def test_init_with_shared_varset_from_first(self):
        rcm8cube1 = cube.DataCube(rcm8_path)
        rcm8cube2 = cube.DataCube(rcm8_path, varset=rcm8cube1.varset)
        assert type(rcm8cube1.varset) is plot.VariableSet
        assert type(rcm8cube2.varset) is plot.VariableSet
        assert rcm8cube1.varset is rcm8cube2.varset

    def test_slice_op(self):
        rcm8cube = cube.DataCube(rcm8_path)
        slc = rcm8cube['eta']
        assert type(slc) is cube.CubeVariable
        assert slc.ndim == 3
        assert type(slc.data) is xr.core.dataarray.DataArray

    def test_slice_op_invalid_name(self):
        rcm8cube = cube.DataCube(rcm8_path)
        with pytest.raises(AttributeError):
            slc = rcm8cube['nonexistentattribute']

    def test_register_section(self):
        rcm8cube = cube.DataCube(rcm8_path)
        rcm8cube.stratigraphy_from('eta')
        rcm8cube.register_section('testsection', section.StrikeSection(y=10))
        assert rcm8cube.sections is rcm8cube.section_set
        assert len(rcm8cube.sections.keys()) == 1
        assert 'testsection' in rcm8cube.sections.keys()

    def test_sections_slice_op(self):
        rcm8cube = cube.DataCube(rcm8_path)
        rcm8cube.stratigraphy_from('eta')
        rcm8cube.register_section('testsection', section.StrikeSection(y=10))
        assert 'testsection' in rcm8cube.sections.keys()
        slc = rcm8cube.sections['testsection']
        assert issubclass(type(slc), section.BaseSection)

    def test_nostratigraphy_default(self):
        rcm8cube = cube.DataCube(rcm8_path)
        assert rcm8cube._knows_stratigraphy is False

    def test_nostratigraphy_default_attribute_derived_variable(self):
        rcm8cube = cube.DataCube(rcm8_path)
        rcm8cube.register_section('testsection', section.StrikeSection(y=10))
        assert rcm8cube._knows_stratigraphy is False
        with pytest.raises(utils.NoStratigraphyError):
            rcm8cube.sections['testsection']['velocity'].as_stratigraphy()

    def test_fixeddatacube_init_varset(self):
        assert type(self.fixeddatacube.varset) is plot.VariableSet

    def test_fixeddatacube_init_data_path(self):
        assert self.fixeddatacube.data_path == rcm8_path

    def test_fixeddatacube_init_dataio(self):
        assert hasattr(self.fixeddatacube, 'dataio')

    def test_fixeddatacube_init_variables(self):
        assert type(self.fixeddatacube.variables) is list

    def test_fixeddatacube_init_plan_set(self):
        assert type(self.fixeddatacube.plan_set) is dict

    def test_fixeddatacube_init_plans(self):
        assert type(self.fixeddatacube.plans) is dict
        assert self.fixeddatacube.plans is self.fixeddatacube.plan_set
        assert len(self.fixeddatacube.plans) == 0

    def test_fixeddatacube_init_section_set(self):
        assert type(self.fixeddatacube.section_set) is dict
        assert len(self.fixeddatacube.section_set) == 0

    def test_fixeddatacube_init_sections(self):
        assert type(self.fixeddatacube.sections) is dict
        assert self.fixeddatacube.sections is self.fixeddatacube.section_set

    def test_fixeddatacube_x(self):
        assert self.fixeddatacube.x.shape == (240,)

    def test_fixeddatacube_X(self):
        assert self.fixeddatacube.X.shape == (120, 240)

    def test_fixeddatacube_y(self):
        assert self.fixeddatacube.y.shape == (120,)

    def test_fixeddatacube_Y(self):
        assert self.fixeddatacube.Y.shape == (120, 240)

    def test_fixeddatacube_z(self):
        assert self.fixeddatacube.z.shape == (51,)
        assert np.all(self.fixeddatacube.z == self.fixeddatacube.t)

    def test_fixeddatacube_Z(self):
        assert self.fixeddatacube.Z.shape == (51, 120, 240)
        assert np.all(self.fixeddatacube.Z == self.fixeddatacube.T)

    def test_fixeddatacube_t(self):
        assert self.fixeddatacube.t.shape == (51,)

    def test_fixeddatacube_T(self):
        assert self.fixeddatacube.T.shape == (51, 120, 240)

    def test_fixeddatacube_H(self):
        assert self.fixeddatacube.H == 51

    def test_fixeddatacube_L(self):
        assert self.fixeddatacube.L == 120

    def test_fixeddatacube_shape(self):
        assert self.fixeddatacube.shape == (51, 120, 240)

    def test_section_no_stratigraphy(self):
        sc = section.StrikeSection(self.fixeddatacube, y=10)
        _ = sc['velocity'][:, 1]
        assert not hasattr(sc, 'strat_attr')
        with pytest.raises(utils.NoStratigraphyError):
            _ = sc.strat_attr
        with pytest.raises(utils.NoStratigraphyError):
            _ = sc['velocity'].as_preserved()


class TestDataCubeWithStratigraphy:

    # create a fixed cube for variable existing, type checks
    fixeddatacube = cube.DataCube(rcm8_path)
    fixeddatacube.stratigraphy_from('eta')  # compute stratigraphy for the cube

    # test setting all the properties / attributes
    def test_fixeddatacube_set_varset(self):
        new_varset = plot.VariableSet()
        self.fixeddatacube.varset = new_varset
        assert hasattr(self.fixeddatacube, 'varset')
        assert type(self.fixeddatacube.varset) is plot.VariableSet
        assert self.fixeddatacube.varset is new_varset

    def test_fixeddatacube_set_varset_bad_type(self):
        with pytest.raises(TypeError):
            self.fixeddatacube.varset = np.zeros(10)

    def test_fixeddatacube_set_data_path(self):
        with pytest.raises(AttributeError):
            self.fixeddatacube.data_path = '/trying/to/change/path.nc'

    def test_fixeddatacube_set_dataio(self):
        with pytest.raises(AttributeError):
            self.fixeddatacube.dataio = 10  # io.NetCDF_IO(rcm8_path)

    def test_fixeddatacube_set_variables_list(self):
        with pytest.raises(AttributeError):
            self.fixeddatacube.variables = ['is', 'a', 'list']

    def test_fixeddatacube_set_variables_dict(self):
        with pytest.raises(AttributeError):
            self.fixeddatacube.variables = {
                'is': True, 'a': True, 'dict': True}

    def test_fixeddatacube_set_plan_set_list(self):
        with pytest.raises(AttributeError):
            self.fixeddatacube.plan_set = ['is', 'a', 'list']

    def test_fixeddatacube_set_plan_set_dict(self):
        with pytest.raises(AttributeError):
            self.fixeddatacube.plan_set = {'is': True, 'a': True, 'dict': True}

    def test_fixeddatacube_set_plans(self):
        with pytest.raises(AttributeError):
            self.fixeddatacube.plans = 10

    def test_fixeddatacube_set_section_set_list(self):
        with pytest.raises(AttributeError):
            self.fixeddatacube.section_set = ['is', 'a', 'list']

    def test_fixeddatacube_set_section_set_dict(self):
        with pytest.raises(AttributeError):
            self.fixeddatacube.section_set = {
                'is': True, 'a': True, 'dict': True}

    def test_fixedset_set_sections(self):
        with pytest.raises(AttributeError):
            self.fixeddatacube.sections = 10

    def test_var_export_frozen(self):
        fdv = self.fixeddatacube['time'].as_frozen()
        assert isinstance(fdv, np.ndarray)
        assert not isinstance(fdv, cube.CubeVariable)
        assert not hasattr(fdv, 'x')

    def test_section_with_stratigraphy(self):
        assert hasattr(self.fixeddatacube, 'strat_attr')
        sc = section.StrikeSection(self.fixeddatacube, y=10)
        assert sc.strat_attr is self.fixeddatacube.strat_attr
        _take = sc['velocity'][:, 1]
        assert _take.shape == (51,)
        assert hasattr(sc, 'strat_attr')
        _take2 = sc['velocity'].as_preserved()
        assert _take2.shape == (51, 240)


class TestStratigraphyCube:

    # create a fixed cube for variable existing, type checks
    fixeddatacube = cube.DataCube(rcm8_path)
    fixedstratigraphycube = cube.StratigraphyCube.from_DataCube(fixeddatacube)

    def test_no_tT_StratigraphyCube(self):
        with pytest.raises(AttributeError):
            _ = self.fixedstratigraphycube.t
        with pytest.raises(AttributeError):
            _ = self.fixedstratigraphycube.T

    def test_export_frozen_variable(self):
        frzn = self.fixedstratigraphycube.export_frozen_variable('time')
        assert frzn.ndim == 3

    def test_var_export_frozen(self):
        fv = self.fixedstratigraphycube['time'].as_frozen()
        assert isinstance(fv, np.ndarray)
        assert not isinstance(fv, cube.CubeVariable)
        assert not hasattr(fv, 'x')
        assert fv.ndim == 3


class TestFrozenStratigraphyCube:

    fixeddatacube = cube.DataCube(rcm8_path)
    fixedstratigraphycube = cube.StratigraphyCube.from_DataCube(fixeddatacube)
    frozenstratigraphycube = fixedstratigraphycube.export_frozen_variable(
        'time')

    def test_types(self):
        assert isinstance(self.frozenstratigraphycube, np.ndarray)

    def test_matches_underlying_data(self):
        assert not self.frozenstratigraphycube is self.fixedstratigraphycube
        frzn_log = self.frozenstratigraphycube[
            ~np.isnan(self.frozenstratigraphycube)]
        fixd_log = self.fixedstratigraphycube['time'][
            ~np.isnan(self.fixedstratigraphycube['time'])]
        assert frzn_log.shape == fixd_log.shape
        assert np.all(fixd_log == frzn_log)
