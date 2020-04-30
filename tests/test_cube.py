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


@pytest.mark.xfail(raises=TypeError, strict=True)
def test_error_init_empty_cube():
    nocube = cube.DataCube()


@pytest.mark.xfail(raises=FileNotFoundError, strict=True)
def test_error_init_bad_path():
    nocube = cube.DataCube('./nonexistent/path.nc')


@pytest.mark.xfail(raises=ValueError, strict=True)
def test_error_init_bad_extension():
    nocube = cube.DataCube('./nonexistent/path.doc')


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


@pytest.mark.xfail(raises=AttributeError, strict=True)
def test_slice_op_invalid_name():
    rcm8cube = cube.DataCube(rcm8_path)
    slc = rcm8cube['nonexistentattribute']


def test_register_section():
    rcm8cube = cube.DataCube(rcm8_path)
    rcm8cube.register_section('testsection', section.StrikeSection(y=10))
    assert rcm8cube.sections is rcm8cube.section_set
    assert len(rcm8cube.sections.keys()) == 1
    assert 'testsection' in rcm8cube.sections.keys()


def test_sections_slice_op():
    rcm8cube = cube.DataCube(rcm8_path)
    rcm8cube.register_section('testsection', section.StrikeSection(y=10))
    assert 'testsection' in rcm8cube.sections.keys()
    slc = rcm8cube.sections['testsection']
    assert issubclass(type(slc), section.BaseSection)

# test plotting routines
# @pytest.mark.mpl_image_compare()
# def test_show_plan():
#     rcm8cube = cube.DataCube(rcm8_path)
#     rcm8cube

#     gui = GUI()
#     return gui.fig


# create a fixed cube for variable existing, type checks
fixedcube = cube.DataCube(rcm8_path)


def test_fixedcube_init_varset():
    assert type(fixedcube.varset) is plot.VariableSet


def test_fixedcube_init_data_path():
    assert fixedcube.data_path == rcm8_path


def test_fixedcube_init_dataio():
    assert hasattr(fixedcube, 'dataio')


def test_fixedcube_init_variables():
    assert type(fixedcube.variables) is list


def test_fixedcube_init_plan_set():
    assert type(fixedcube.plan_set) is dict


def test_fixedcube_init_plans():
    assert type(fixedcube.plans) is dict
    assert fixedcube.plans is fixedcube.plan_set
    assert len(fixedcube.plans) == 0


def test_fixedcube_init_section_set():
    assert type(fixedcube.section_set) is dict
    assert len(fixedcube.section_set) == 0


def test_fixedcube_init_sections():
    assert type(fixedcube.sections) is dict
    assert fixedcube.sections is fixedcube.section_set


# compute stratigraphy for the cube
fixedcube.stratigraphy_from('eta')


def test_fixedcube_init_preserved_index():
    assert type(fixedcube.preserved_index) is np.ndarray


def test_fixedcube_init_preserved_voxel_count():
    assert type(fixedcube.preserved_voxel_count) is np.ndarray


# test setting all the properties / attributes

@pytest.mark.xfail(raises=AttributeError, strict=True)
def test_fixedcube_set_preserved_index():
    fixedcube.preserved_index = 10


@pytest.mark.xfail(raises=AttributeError, strict=True)
def test_fixedcube_set_preserved_voxel_count():
    fixedcube.preserved_voxel_count = 10


def test_fixedcube_set_varset():
    new_varset = plot.VariableSet()
    fixedcube.varset = new_varset
    assert hasattr(fixedcube, 'varset')
    assert type(fixedcube.varset) is plot.VariableSet
    assert fixedcube.varset is new_varset


@pytest.mark.xfail(raises=TypeError, strict=True)
def test_fixedcube_set_varset_bad_type():
    fixedcube.varset = np.zeros(10)


@pytest.mark.xfail(raises=AttributeError, strict=True)
def test_fixedcube_set_data_path():
    fixedcube.data_path = '/trying/to/change/path.nc'


@pytest.mark.xfail(raises=AttributeError, strict=True)
def test_fixedcube_set_dataio():
    fixedcube.dataio = 10  # io.NetCDF_IO(rcm8_path)


@pytest.mark.xfail(raises=AttributeError, strict=True)
def test_fixedcube_set_variables_list():
    fixedcube.variables = ['is', 'a', 'list']


@pytest.mark.xfail(raises=AttributeError, strict=True)
def test_fixedcube_set_variables_dict():
    fixedcube.variables = {'is': True, 'a': True, 'dict': True}


@pytest.mark.xfail(raises=AttributeError, strict=True)
def test_fixedcube_set_plan_set_list():
    fixedcube.plan_set = ['is', 'a', 'list']


@pytest.mark.xfail(raises=AttributeError, strict=True)
def test_fixedcube_set_plan_set_dict():
    fixedcube.plan_set = {'is': True, 'a': True, 'dict': True}


@pytest.mark.xfail(raises=AttributeError, strict=True)
def test_fixedcube_set_plans():
    fixedcube.plans = 10


@pytest.mark.xfail(raises=AttributeError, strict=True)
def test_fixedcube_set_section_set_list():
    fixedcube.section_set = ['is', 'a', 'list']


@pytest.mark.xfail(raises=AttributeError, strict=True)
def test_fixedcube_set_section_set_dict():
    fixedcube.section_set = {'is': True, 'a': True, 'dict': True}


@pytest.mark.xfail(raises=AttributeError, strict=True)
def test_fixedset_set_sections():
    fixedcube.sections = 10
