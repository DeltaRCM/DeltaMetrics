import pytest
import re

import numpy as np
import xarray as xr

from deltametrics import cube

from deltametrics import plot
from deltametrics import section
from deltametrics import utils
from deltametrics.sample_data import _get_golf_path, _get_rcm8_path, _get_landsat_path


rcm8_path = _get_rcm8_path()
golf_path = _get_golf_path()
hdf_path = _get_landsat_path()


class TestDataCubeNoStratigraphy:

    # create a fixed cube for variable existing, type checks
    fixeddatacube = cube.DataCube(golf_path)

    fdc_shape = fixeddatacube.shape

    def test_init_cube_from_path_rcm8(self):
        golf = cube.DataCube(golf_path)
        assert golf._data_path == golf_path
        assert golf.dataio.type == 'netcdf'
        assert golf._plan_set == {}
        assert golf._section_set == {}
        assert type(golf.varset) is plot.VariableSet

    def test_error_init_empty_cube(self):
        with pytest.raises(TypeError):
            _ = cube.DataCube()

    def test_error_init_bad_path(self):
        with pytest.raises(FileNotFoundError):
            _ = cube.DataCube('./nonexistent/path.nc')

    def test_error_init_bad_extension(self):
        with pytest.raises(ValueError):
            _ = cube.DataCube('./nonexistent/path.doc')

    def test_stratigraphy_from_eta(self):
        golf0 = cube.DataCube(golf_path)
        golf1 = cube.DataCube(golf_path)
        golf0.stratigraphy_from('eta')
        assert golf0._knows_stratigraphy is True
        assert golf1._knows_stratigraphy is False

    def test_init_cube_stratigraphy_argument(self):
        golf = cube.DataCube(golf_path, stratigraphy_from='eta')
        assert golf._knows_stratigraphy is True

    def test_stratigraphy_from_default_noargument(self):
        golf = cube.DataCube(golf_path)
        golf.stratigraphy_from()
        assert golf._knows_stratigraphy is True

    def test_init_with_shared_varset_prior(self):
        shared_varset = plot.VariableSet()
        golf1 = cube.DataCube(golf_path, varset=shared_varset)
        golf2 = cube.DataCube(golf_path, varset=shared_varset)
        assert type(golf1.varset) is plot.VariableSet
        assert type(golf2.varset) is plot.VariableSet
        assert golf1.varset is shared_varset
        assert golf1.varset is golf2.varset

    def test_init_with_shared_varset_from_first(self):
        golf1 = cube.DataCube(golf_path)
        golf2 = cube.DataCube(golf_path, varset=golf1.varset)
        assert type(golf1.varset) is plot.VariableSet
        assert type(golf2.varset) is plot.VariableSet
        assert golf1.varset is golf2.varset

    def test_slice_op(self):
        golf = cube.DataCube(golf_path)
        slc = golf['eta']
        assert type(slc) is xr.core.dataarray.DataArray
        assert slc.ndim == 3
        assert type(slc.values) is np.ndarray

    def test_slice_op_invalid_name(self):
        golf = cube.DataCube(golf_path)
        with pytest.raises(AttributeError):
            _ = golf['nonexistentattribute']

    def test_register_section(self):
        golf = cube.DataCube(golf_path)
        golf.stratigraphy_from('eta', dz=0.1)
        golf.register_section(
            'testsection', section.StrikeSection(distance_idx=10))
        assert golf.sections is golf.section_set
        assert len(golf.sections.keys()) == 1
        assert 'testsection' in golf.sections.keys()

    def test_sections_slice_op(self):
        golf = cube.DataCube(golf_path)
        golf.stratigraphy_from('eta', dz=0.1)
        golf.register_section(
            'testsection', section.StrikeSection(distance_idx=10))
        assert 'testsection' in golf.sections.keys()
        slc = golf.sections['testsection']
        assert issubclass(type(slc), section.BaseSection)

    def test_nostratigraphy_default(self):
        golf = cube.DataCube(golf_path)
        assert golf._knows_stratigraphy is False

    def test_nostratigraphy_default_attribute_derived_variable(self):
        golf = cube.DataCube(golf_path)
        golf.register_section(
            'testsection', section.StrikeSection(distance_idx=10))
        assert golf._knows_stratigraphy is False
        with pytest.raises(utils.NoStratigraphyError):
            golf.sections['testsection']['velocity'].strat.as_stratigraphy()

    def test_fixeddatacube_init_varset(self):
        assert type(self.fixeddatacube.varset) is plot.VariableSet

    def test_fixeddatacube_init_data_path(self):
        assert self.fixeddatacube.data_path == golf_path

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

    def test_metadata_present(self):
        assert self.fixeddatacube.meta is self.fixeddatacube._dataio.meta

    def test_fixeddatacube_dim1_coords(self):
        assert self.fixeddatacube.dim1_coords.shape == (self.fdc_shape[1],)

    def test_fixeddatacube_dim2_coords(self):
        assert self.fixeddatacube.dim2_coords.shape == (self.fdc_shape[2],)

    def test_fixeddatacube_z(self):
        assert self.fixeddatacube.z.shape == (self.fdc_shape[0],)
        assert np.all(self.fixeddatacube.z == self.fixeddatacube.t)

    def test_fixeddatacube_Z(self):
        assert self.fixeddatacube.Z.shape == self.fdc_shape
        assert np.all(self.fixeddatacube.Z == self.fixeddatacube.T)

    def test_fixeddatacube_t(self):
        assert self.fixeddatacube.t.shape == (self.fdc_shape[0],)

    def test_fixeddatacube_T(self):
        assert self.fixeddatacube.T.shape == self.fdc_shape

    def test_fixeddatacube_H(self):
        assert self.fixeddatacube.H == self.fdc_shape[0]

    def test_fixeddatacube_L(self):
        assert self.fixeddatacube.L == self.fdc_shape[1]

    def test_fixeddatacube_shape(self):
        assert self.fixeddatacube.shape == self.fdc_shape

    def test_section_no_stratigraphy(self):
        sc = section.StrikeSection(self.fixeddatacube, distance_idx=10)
        _ = sc['velocity'][:, 1]
        assert not hasattr(sc, 'strat_attr')
        with pytest.raises(utils.NoStratigraphyError):
            _ = sc.strat_attr
        with pytest.raises(utils.NoStratigraphyError):
            _ = sc['velocity'].strat.as_preserved()


class TestDataCubeWithStratigraphy:

    # create a fixed cube for variable existing, type checks
    fixeddatacube = cube.DataCube(golf_path)
    fixeddatacube.stratigraphy_from('eta', dz=0.1)  # compute stratigraphy for the cube

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
            self.fixeddatacube.dataio = 10  # io.NetCDF_IO(golf_path)

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

    def test_export_frozen_variable(self):
        frzn = self.fixeddatacube.export_frozen_variable('velocity')
        assert frzn.ndim == 3

    def test_section_with_stratigraphy(self):
        assert hasattr(self.fixeddatacube, 'strat_attr')
        sc = section.StrikeSection(self.fixeddatacube, distance_idx=10)
        assert sc.strat_attr is self.fixeddatacube.strat_attr
        _take = sc['velocity'][:, 1]
        assert _take.shape == (self.fixeddatacube.shape[0],)
        assert hasattr(sc, 'strat_attr')
        _take2 = sc['velocity'].strat.as_preserved()
        assert _take2.shape == (self.fixeddatacube.shape[0], self.fixeddatacube.shape[2])


class TestStratigraphyCube:

    # create a fixed cube for variable existing, type checks
    fixeddatacube = cube.DataCube(golf_path)
    fixedstratigraphycube = cube.StratigraphyCube.from_DataCube(
        fixeddatacube, dz=0.1)

    def test_no_tT_StratigraphyCube(self):
        with pytest.raises(AttributeError):
            _ = self.fixedstratigraphycube.t
        with pytest.raises(AttributeError):
            _ = self.fixedstratigraphycube.T

    def test_export_frozen_variable(self):
        frzn = self.fixedstratigraphycube.export_frozen_variable('time')
        assert frzn.ndim == 3


class TestFrozenStratigraphyCube:

    fixeddatacube = cube.DataCube(golf_path)
    fixedstratigraphycube = cube.StratigraphyCube.from_DataCube(
        fixeddatacube, dz=0.1)
    frozenstratigraphycube = fixedstratigraphycube.export_frozen_variable(
        'time')

    def test_types(self):
        assert isinstance(self.frozenstratigraphycube,
                          xr.core.dataarray.DataArray)

    def test_matches_underlying_data(self):
        assert not (self.frozenstratigraphycube is self.fixedstratigraphycube)
        frzn_log = self.frozenstratigraphycube.values[
            ~np.isnan(self.frozenstratigraphycube.values)]
        fixd_log = self.fixedstratigraphycube['time'].values[
            ~np.isnan(self.fixedstratigraphycube['time'].values)]
        assert frzn_log.shape == fixd_log.shape
        assert np.all(fixd_log == frzn_log)


class TestLegacyPyDeltaRCMCube:

    def test_init_cube_from_path_rcm8(self):
        with pytest.warns(UserWarning) as record:
            rcm8cube = cube.DataCube(rcm8_path)
        assert rcm8cube._data_path == rcm8_path
        assert rcm8cube.dataio.type == 'netcdf'
        assert rcm8cube._plan_set == {}
        assert rcm8cube._section_set == {}
        assert type(rcm8cube.varset) is plot.VariableSet

        # check that two warnings were raised
        assert re.match(
            r'Coordinates for "time", .*',
            record[0].message.args[0]
            )
        assert re.match(
            r'No associated metadata .*',
            record[1].message.args[0])

    def test_warning_netcdf_no_metadata(self):
        with pytest.warns(UserWarning, match=r'No associated metadata'):
            _ = cube.DataCube(rcm8_path)

    def test_metadata_none_nometa(self):
        with pytest.warns(UserWarning):
            rcm8cube = cube.DataCube(rcm8_path)
        assert rcm8cube.meta is None


class TestLandsatCube:

    with pytest.warns(UserWarning, match=r'No associated metadata'):
        landsatcube = cube.DataCube(hdf_path)

    def test_init_cube_from_path_hdf5(self):
        with pytest.warns(UserWarning, match=r'No associated metadata'):
            hdfcube = cube.DataCube(hdf_path)
        assert hdfcube._data_path == hdf_path
        assert hdfcube.dataio.type == 'hdf5'
        assert hdfcube._plan_set == {}
        assert hdfcube._section_set == {}
        assert type(hdfcube.varset) is plot.VariableSet

    def test_read_Blue_intomemory(self):
        assert self.landsatcube._dataio._in_memory_data == {}
        assert self.landsatcube.variables == ['Blue', 'Green', 'NIR', 'Red']
        assert len(self.landsatcube.variables) == 4

        self.landsatcube.read('Blue')
        assert len(self.landsatcube.dataio._in_memory_data) == 1

    def test_read_all_intomemory(self):
        assert self.landsatcube.variables == ['Blue', 'Green', 'NIR', 'Red']
        assert len(self.landsatcube.variables) == 4

        self.landsatcube.read(True)
        assert len(self.landsatcube.dataio._in_memory_data) == 4

    def test_read_invalid(self):
        with pytest.raises(TypeError):
            self.landsatcube.read(5)

    def test_get_coords(self):
        assert self.landsatcube.coords == ['time', 'x', 'y']
        assert self.landsatcube._coords == ['time', 'x', 'y']
