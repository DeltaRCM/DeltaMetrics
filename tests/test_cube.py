import pytest

import sys
import os

import numpy as np

from deltametrics import cube

from deltametrics import plot
from deltametrics import section

# initialize a cube directly from path, rather than using sample_data.py
rcm8_path = os.path.join(os.path.dirname(__file__), '..', 'deltametrics',
                         'sample_data', 'files', 'pyDeltaRCM_Output_8.nc')


def test_init_cube_from_path_rcm8():
    rcm8cube = cube.DataCube(rcm8_path)
    assert rcm8cube._data_path == rcm8_path
    assert rcm8cube.dataio.type == 'netcdf'
    assert rcm8cube._plan_set == {}
    assert rcm8cube._section_set == {}
    assert type(rcm8cube.varset) is plot.VariableSet


def test_error_init_empty_cube():
    with pytest.raises(TypeError):
        nocube = cube.DataCube()


def test_error_init_bad_path():
    with pytest.raises(FileNotFoundError):
        nocube = cube.DataCube('./nonexistent/path.nc')


def test_error_init_bad_extension():
    with pytest.raises(ValueError):
        nocube = cube.DataCube('./nonexistent/path.doc')


def test_stratigraphy_from_eta():
    rcm8cube = cube.DataCube(rcm8_path)
    rcm8cube.stratigraphy_from('eta')
    assert rcm8cube._knows_stratigraphy is True


def test_init_cube_stratigraphy_argument():
    rcm8cube = cube.DataCube(rcm8_path, stratigraphy_from='eta')
    assert rcm8cube._knows_stratigraphy is True


def test_stratigraphy_from_default_noargument():
    rcm8cube = cube.DataCube(rcm8_path)
    rcm8cube.stratigraphy_from()
    assert rcm8cube._knows_stratigraphy is True


def test_init_with_shared_varset_prior():
    shared_varset = plot.VariableSet()
    rcm8cube1 = cube.DataCube(rcm8_path, varset=shared_varset)
    rcm8cube2 = cube.DataCube(rcm8_path, varset=shared_varset)
    assert type(rcm8cube1.varset) is plot.VariableSet
    assert type(rcm8cube2.varset) is plot.VariableSet
    assert rcm8cube1.varset is shared_varset
    assert rcm8cube1.varset is rcm8cube2.varset


def test_init_with_shared_varset_from_first():
    rcm8cube1 = cube.DataCube(rcm8_path)
    rcm8cube2 = cube.DataCube(rcm8_path, varset=rcm8cube1.varset)
    assert type(rcm8cube1.varset) is plot.VariableSet
    assert type(rcm8cube2.varset) is plot.VariableSet
    assert rcm8cube1.varset is rcm8cube2.varset


def test_slice_op():
    rcm8cube = cube.DataCube(rcm8_path)
    slc = rcm8cube['eta']
    assert type(slc) is cube.CubeVariable
    assert slc.ndim == 3
    assert type(slc.base) is np.ndarray


def test_slice_op_invalid_name():
    rcm8cube = cube.DataCube(rcm8_path)
    with pytest.raises(AttributeError):
        slc = rcm8cube['nonexistentattribute']


def test_register_section():
    rcm8cube = cube.DataCube(rcm8_path)
    rcm8cube.stratigraphy_from('eta')
    rcm8cube.register_section('testsection', section.StrikeSection(y=10))
    assert rcm8cube.sections is rcm8cube.section_set
    assert len(rcm8cube.sections.keys()) == 1
    assert 'testsection' in rcm8cube.sections.keys()


def test_sections_slice_op():
    rcm8cube = cube.DataCube(rcm8_path)
    rcm8cube.stratigraphy_from('eta')
    rcm8cube.register_section('testsection', section.StrikeSection(y=10))
    assert 'testsection' in rcm8cube.sections.keys()
    slc = rcm8cube.sections['testsection']
    assert issubclass(type(slc), section.BaseSection)


def test_nostratigraphy_default():
    rcm8cube = cube.DataCube(rcm8_path)
    assert rcm8cube._knows_stratigraphy is False


def test_nostratigraphy_default_attribute_derived_variable():
    rcm8cube = cube.DataCube(rcm8_path)
    rcm8cube.register_section('testsection', section.StrikeSection(y=10))
    assert rcm8cube._knows_stratigraphy is False
    with pytest.raises(AttributeError, match=r'No preservation information.'):
        rcm8cube.sections['testsection']['velocity'].as_stratigraphy()


# create a fixed cube for variable existing, type checks
fixeddatacube = cube.DataCube(rcm8_path)


def test_fixeddatacube_init_varset():
    assert type(fixeddatacube.varset) is plot.VariableSet


def test_fixeddatacube_init_data_path():
    assert fixeddatacube.data_path == rcm8_path


def test_fixeddatacube_init_dataio():
    assert hasattr(fixeddatacube, 'dataio')


def test_fixeddatacube_init_variables():
    assert type(fixeddatacube.variables) is list


def test_fixeddatacube_init_plan_set():
    assert type(fixeddatacube.plan_set) is dict


def test_fixeddatacube_init_plans():
    assert type(fixeddatacube.plans) is dict
    assert fixeddatacube.plans is fixeddatacube.plan_set
    assert len(fixeddatacube.plans) == 0


def test_fixeddatacube_init_section_set():
    assert type(fixeddatacube.section_set) is dict
    assert len(fixeddatacube.section_set) == 0


def test_fixeddatacube_init_sections():
    assert type(fixeddatacube.sections) is dict
    assert fixeddatacube.sections is fixeddatacube.section_set


# compute stratigraphy for the cube
fixeddatacube.stratigraphy_from('eta')


# test setting all the properties / attributes
def test_fixeddatacube_set_varset():
    new_varset = plot.VariableSet()
    fixeddatacube.varset = new_varset
    assert hasattr(fixeddatacube, 'varset')
    assert type(fixeddatacube.varset) is plot.VariableSet
    assert fixeddatacube.varset is new_varset


def test_fixeddatacube_set_varset_bad_type():
    with pytest.raises(TypeError):
        fixeddatacube.varset = np.zeros(10)


def test_fixeddatacube_set_data_path():
    with pytest.raises(AttributeError):
        fixeddatacube.data_path = '/trying/to/change/path.nc'


def test_fixeddatacube_set_dataio():
    with pytest.raises(AttributeError):
        fixeddatacube.dataio = 10  # io.NetCDF_IO(rcm8_path)


def test_fixeddatacube_set_variables_list():
    with pytest.raises(AttributeError):
        fixeddatacube.variables = ['is', 'a', 'list']


def test_fixeddatacube_set_variables_dict():
    with pytest.raises(AttributeError):
        fixeddatacube.variables = {'is': True, 'a': True, 'dict': True}


def test_fixeddatacube_set_plan_set_list():
    with pytest.raises(AttributeError):
        fixeddatacube.plan_set = ['is', 'a', 'list']


def test_fixeddatacube_set_plan_set_dict():
    with pytest.raises(AttributeError):
        fixeddatacube.plan_set = {'is': True, 'a': True, 'dict': True}


def test_fixeddatacube_set_plans():
    with pytest.raises(AttributeError):
        fixeddatacube.plans = 10


def test_fixeddatacube_set_section_set_list():
    with pytest.raises(AttributeError):
        fixeddatacube.section_set = ['is', 'a', 'list']

def test_fixeddatacube_set_section_set_dict():
    with pytest.raises(AttributeError):
        fixeddatacube.section_set = {'is': True, 'a': True, 'dict': True}


def test_fixedset_set_sections():
    with pytest.raises(AttributeError):
        fixeddatacube.sections = 10


fixedstratigraphycube = cube.StratigraphyCube.from_DataCube(fixeddatacube)

def test_no_tT_StratigraphyCube():
    with pytest.raises(AttributeError):
        _ = fixedstratigraphycube.t
    with pytest.raises(AttributeError):
        _ = fixedstratigraphycube.T
