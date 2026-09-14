import unittest.mock as mock

import numpy as np
import pytest
import xarray as xr

from sandplover.cube import DataCube
from sandplover.cube import StratigraphyCube
from sandplover.plan import BasePlanform
from sandplover.plan import Planform
from sandplover.plot import VariableSet
from sandplover.sample_data.sample_data import _get_aeolian_path
from sandplover.sample_data.sample_data import _get_golf_path
from sandplover.sample_data.sample_data import _get_golf_sandsuet_path
from sandplover.sample_data.sample_data import _get_landsat_path
from sandplover.sample_data.sample_data import _get_rcm8_path
from sandplover.sample_data.sample_data import rcm8
from sandplover.section import BaseSection
from sandplover.section import StrikeSection
from sandplover.utils import NoStratigraphyError

golf_path = _get_golf_sandsuet_path()
aeolian_path = _get_aeolian_path()
hdf_path = _get_landsat_path()


@mock.patch("sandplover.cube.NetCDFIO")
class TestDataCubeInitializationArguments:

    def test_initializing_without_argument_nc(self, mock_netcdfio):
        # cube = DataCube(tdb12_path)
        # assert cube.aux is None
        ## NO SAMPLE DATA AVAILABLE TO TEST
        pass

    def test_initializing_without_argument_warns_autodetect(self, mock_netcdfio):
        with pytest.raises(Exception):
            # the functions following instantiation will error out, so just
            # check that argument was passed to io
            golf = DataCube(golf_path)
        mock_netcdfio.assert_called_once_with(data_path=mock.ANY, auxdata_path=None)

    def test_initializing_with_argument(self, mock_netcdfio):
        with pytest.raises(Exception):
            # the functions following instantiation will error out, so just
            # check that argument was passed to io
            golf = DataCube(golf_path, auxdata="meta")
        mock_netcdfio.assert_called_once_with(data_path=mock.ANY, auxdata_path="meta")


class TestDataCubeNoStratigraphy:

    def test_init_cube_from_path_golf(self):
        golf = DataCube(golf_path)
        assert golf._data_path == golf_path
        assert golf.dataio.io_type == "file"
        assert golf._planform_set == {}
        assert golf._section_set == {}
        assert type(golf.varset) is VariableSet

    def test_error_init_empty_cube(self):
        with pytest.raises(TypeError):
            _ = DataCube()

    def test_error_init_bad_path(self):
        with pytest.raises(FileNotFoundError):
            _ = DataCube("./nonexistent/path.nc")

    def test_error_init_bad_extension(self):
        with pytest.raises(FileNotFoundError):
            _ = DataCube("./nonexistent/path.doc")

    def test_error_init_bad_type(self):
        with pytest.raises(TypeError):
            _ = DataCube(9)

    def test_init_with_shared_varset_prior(self):
        shared_varset = VariableSet()
        golf1 = DataCube(golf_path, varset=shared_varset)
        golf2 = DataCube(golf_path, varset=shared_varset)
        assert type(golf1.varset) is VariableSet
        assert type(golf2.varset) is VariableSet
        assert golf1.varset is shared_varset
        assert golf1.varset is golf2.varset

    def test_init_with_shared_varset_from_first(self):
        golf1 = DataCube(golf_path)
        golf2 = DataCube(golf_path, varset=golf1.varset)
        assert type(golf1.varset) is VariableSet
        assert type(golf2.varset) is VariableSet
        assert golf1.varset is golf2.varset

    def test_slice_op(self):
        golf = DataCube(golf_path)
        slc = golf["eta"]
        assert type(slc) is xr.core.dataarray.DataArray
        assert slc.ndim == 3
        assert type(slc.values) is np.ndarray

    def test_slice_op_invalid_name(self):
        golf = DataCube(golf_path)
        with pytest.raises(AttributeError):
            _ = golf["nonexistentattribute"]

    def test_register_section(self):
        golf = DataCube(golf_path)
        golf.register_section("testsection", StrikeSection(distance_idx=10))
        assert golf.sections is golf.section_set
        assert len(golf.sections) == 1
        assert "testsection" in golf.sections
        with pytest.raises(TypeError, match=r"`SectionInstance` .*"):
            golf.register_section("fail1", "astring")
        with pytest.raises(TypeError, match=r"`SectionInstance` .*"):
            golf.register_section("fail2", 22)
        with pytest.raises(TypeError, match=r"`name` .*"):
            golf.register_section(22, StrikeSection(distance_idx=10))

    def test_sections_slice_op(self):
        golf = DataCube(golf_path)
        golf.register_section("testsection", StrikeSection(distance_idx=10))
        assert "testsection" in golf.sections
        slc = golf.sections["testsection"]
        assert issubclass(type(slc), BaseSection)

    def test_register_planform(self):
        golf = DataCube(golf_path)
        golf.register_planform("testplanform", Planform(idx=10))
        assert golf.planforms is golf.planform_set
        assert len(golf.planforms) == 1
        assert "testplanform" in golf.planforms
        with pytest.raises(TypeError, match=r"`PlanformInstance` .*"):
            golf.register_planform("fail1", "astring")
        with pytest.raises(TypeError, match=r"`PlanformInstance` .*"):
            golf.register_planform("fail2", 22)
        with pytest.raises(TypeError, match=r"`name` .*"):
            golf.register_planform(22, Planform(idx=10))
        returnedplanform = golf.register_planform(
            "returnedplanform", Planform(idx=10), return_planform=True
        )
        assert returnedplanform.name == "returnedplanform"

    def test_register_plan_legacy_method(self):
        """This tests the shorthand named version."""
        golf = DataCube(golf_path)
        golf.register_plan("testplanform", Planform(idx=10))
        assert golf.planforms is golf.planform_set
        assert len(golf.planforms) == 1
        assert "testplanform" in golf.planforms

    def test_planforms_slice_op(self):
        golf = DataCube(golf_path)
        golf.register_planform("testplanform", Planform(idx=10))
        assert "testplanform" in golf.planforms
        slc = golf.planforms["testplanform"]
        assert issubclass(type(slc), BasePlanform)

    def test_nostratigraphy_default(self):
        golf = DataCube(golf_path)
        assert golf._knows_stratigraphy is False

    def test_nostratigraphy_default_attribute_derived_variable(self):
        golf = DataCube(golf_path)
        golf.register_section("testsection", StrikeSection(distance_idx=10))
        assert golf._knows_stratigraphy is False
        with pytest.raises(NoStratigraphyError):
            golf.sections["testsection"]["velocity"].strat.as_stratigraphy()

    def test_register_variable(self):
        golf = DataCube(golf_path)
        golf.register_variable("testvar", np.zeros(golf.shape))
        assert "testvar" in golf.variables
        assert np.all(golf["testvar"].shape == golf.shape)

    def test_register_variable_bad_inputs(self):
        golf = DataCube(golf_path)
        with pytest.raises(ValueError, match=r"Input 'data' was incorrect"):
            golf.register_variable("testvar", np.zeros((10, 10, 10)))
        with pytest.raises(ValueError, match=r"Input 'data' was incorrect"):
            golf.register_variable("testvar", np.zeros((10, 10)))
        with pytest.raises(TypeError, match=r"Input 'name' was not"):
            golf.register_variable(33, "name")

    def test_fixeddatacube_init_varset(self):
        fixeddatacube = DataCube(golf_path)
        assert type(fixeddatacube.varset) is VariableSet

    def test_fixeddatacube_init_data_path(self):
        fixeddatacube = DataCube(golf_path)
        assert fixeddatacube.data_path == golf_path

    def test_fixeddatacube_init_dataio(self):
        fixeddatacube = DataCube(golf_path)
        assert hasattr(fixeddatacube, "dataio")

    def test_fixeddatacube_init_variables(self):
        fixeddatacube = DataCube(golf_path)
        assert type(fixeddatacube.variables) is list

    def test_fixeddatacube_init_planform_set(self):
        fixeddatacube = DataCube(golf_path)
        assert type(fixeddatacube.plan_set) is dict

    def test_fixeddatacube_init_plans(self):
        fixeddatacube = DataCube(golf_path)
        assert type(fixeddatacube.plans) is dict
        assert fixeddatacube.plans is fixeddatacube.plan_set
        assert len(fixeddatacube.plans) == 0

    def test_fixeddatacube_init_section_set(self):
        fixeddatacube = DataCube(golf_path)
        assert type(fixeddatacube.section_set) is dict
        assert len(fixeddatacube.section_set) == 0

    def test_fixeddatacube_init_sections(self):
        fixeddatacube = DataCube(golf_path)
        assert type(fixeddatacube.sections) is dict
        assert fixeddatacube.sections is fixeddatacube.section_set

    def test_auxadata_present(self):
        fixeddatacube = DataCube(golf_path)
        assert fixeddatacube.aux is fixeddatacube._dataio.aux
        with pytest.warns(DeprecationWarning, match=r"The `meta` property"):
            fixeddatacube.meta

    def test_fixeddatacube_dim1_coords(self):
        fixeddatacube = DataCube(golf_path)
        fdc_shape = fixeddatacube.shape
        assert fixeddatacube.dim1_coords.shape == (fdc_shape[1],)

    def test_fixeddatacube_dim2_coords(self):
        fixeddatacube = DataCube(golf_path)
        fdc_shape = fixeddatacube.shape
        assert fixeddatacube.dim2_coords.shape == (fdc_shape[2],)

    def test_fixeddatacube_z(self):
        fixeddatacube = DataCube(golf_path)
        fdc_shape = fixeddatacube.shape
        assert fixeddatacube.z.shape == (fdc_shape[0],)
        assert np.all(fixeddatacube.z == fixeddatacube.t)

    def test_fixeddatacube_Z(self):
        fixeddatacube = DataCube(golf_path)
        fdc_shape = fixeddatacube.shape
        assert fixeddatacube.Z.shape == fdc_shape
        assert np.all(fixeddatacube.Z == fixeddatacube.T)

    def test_fixeddatacube_t(self):
        fixeddatacube = DataCube(golf_path)
        fdc_shape = fixeddatacube.shape
        assert fixeddatacube.t.shape == (fdc_shape[0],)

    def test_fixeddatacube_T(self):
        fixeddatacube = DataCube(golf_path)
        fdc_shape = fixeddatacube.shape
        assert fixeddatacube.T.shape == fdc_shape

    def test_fixeddatacube_H(self):
        fixeddatacube = DataCube(golf_path)
        fdc_shape = fixeddatacube.shape
        assert fixeddatacube.H == fdc_shape[0]

    def test_fixeddatacube_L(self):
        fixeddatacube = DataCube(golf_path)
        fdc_shape = fixeddatacube.shape
        assert fixeddatacube.L == fdc_shape[1]

    def test_fixeddatacube_shape(self):
        fixeddatacube = DataCube(golf_path)
        assert isinstance(fixeddatacube.shape, tuple)

    def test_section_no_stratigraphy(self):
        fixeddatacube = DataCube(golf_path)
        sc = StrikeSection(fixeddatacube, distance_idx=10)
        _ = sc["velocity"][:, 1]
        assert not hasattr(sc, "strat_attr")
        with pytest.raises(NoStratigraphyError):
            _ = sc.strat_attr
        with pytest.raises(NoStratigraphyError):
            _ = sc["velocity"].strat.as_preserved()

    def test_show_section_mocked_BaseSection_show(self):
        golf = DataCube(golf_path)
        golf.register_section("displaysection", StrikeSection(distance_idx=10))
        golf.sections["displaysection"].show = mock.MagicMock()
        mocked = golf.sections["displaysection"].show
        # no arguments is an error
        with pytest.raises(TypeError, match=r".* missing 2 .*"):
            golf.show_section()
        # one argument is an error
        with pytest.raises(TypeError, match=r".* missing 1 .*"):
            golf.show_section("displaysection")
        # three arguments is an error
        with pytest.raises(TypeError, match=r".* takes 3 .*"):
            golf.show_section("one", "two", "three")
        # two arguments passes to BaseSection.show()
        golf.show_section("displaysection", "eta")
        assert mocked.call_count == 1
        # kwargs should be passed along to BaseSection.show
        golf.show_section("displaysection", "eta", ax=100)
        assert mocked.call_count == 2
        mocked.assert_called_with("eta", ax=100)
        # first arg must be a string
        with pytest.raises(TypeError, match=r"`name` was not .*"):
            golf.show_section(1, "two")

    def test_show_planform_mocked_Planform_show(self):
        golf = DataCube(golf_path)
        golf.register_planform("displayplan", Planform(idx=-1))
        golf.planforms["displayplan"].show = mock.MagicMock()
        mocked = golf.planforms["displayplan"].show
        # no arguments is an error
        with pytest.raises(TypeError, match=r".* missing 2 .*"):
            golf.show_planform()
        # one argument is an error
        with pytest.raises(TypeError, match=r".* missing 1 .*"):
            golf.show_planform("displayplan")
        # three arguments is an error
        with pytest.raises(TypeError, match=r".* takes 3 .*"):
            golf.show_planform("one", "two", "three")
        # two arguments passes to BaseSection.show()
        golf.show_planform("displayplan", "eta")
        assert mocked.call_count == 1
        # kwargs should be passed along to BaseSection.show
        golf.show_planform("displayplan", "eta", ax=100)
        assert mocked.call_count == 2
        mocked.assert_called_with("eta", ax=100)
        # first arg must be a string
        with pytest.raises(TypeError, match=r"`name` was not .*"):
            golf.show_planform(1, "two")

    def test_extent(self):
        golf = DataCube(golf_path)
        assert len(golf.extent) == 4
        assert isinstance(golf.extent, list)
        assert isinstance(golf.extent[0], float)

    def test_extent_flipud(self):
        golf = DataCube(golf_path)
        assert len(golf.extent_flipud) == 4
        assert isinstance(golf.extent_flipud, list)
        assert isinstance(golf.extent_flipud[0], float)

    def test_access_groups_manually(self):
        golf = DataCube(golf_path, auxdata="auxdata")
        assert np.all(golf.dataio.dataset["auxdata"]["H_SL"] == golf.aux["H_SL"])

    def test_set_aux_data(self):
        golf = DataCube(golf_path)
        golf.set_aux("auxdata")
        assert np.all(golf.dataio.dataset["auxdata"]["H_SL"] == golf.aux["H_SL"])


class TestDataCubeWithStratigraphy:
    def test_stratigraphy_from_eta(self):
        golf0 = DataCube(golf_path)
        golf1 = DataCube(golf_path)
        golf0.stratigraphy_from("eta")
        assert golf0._knows_stratigraphy is True
        assert golf1._knows_stratigraphy is False

    def test_init_cube_stratigraphy_argument(self):
        golf = DataCube(golf_path, stratigraphy_from="eta")
        assert golf._knows_stratigraphy is True

    def test_stratigraphy_from_default_noargument(self):
        golf = DataCube(golf_path)
        golf.stratigraphy_from()
        assert golf._knows_stratigraphy is True

    # test setting all the properties / attributes
    def test_fixeddatacube_set_varset(self):
        # create a fixed cube for variable existing, type checks
        fixeddatacube = DataCube(golf_path)
        fixeddatacube.stratigraphy_from(
            "eta", dz=0.1
        )  # compute stratigraphy for the cube

        new_varset = VariableSet()
        fixeddatacube.varset = new_varset
        assert hasattr(fixeddatacube, "varset")
        assert type(fixeddatacube.varset) is VariableSet
        assert fixeddatacube.varset is new_varset

    def test_fixeddatacube_set_varset_bad_type(self):
        fixeddatacube = DataCube(golf_path)
        fixeddatacube.stratigraphy_from("eta", dz=0.1)
        with pytest.raises(TypeError):
            fixeddatacube.varset = np.zeros(10)

    def test_fixeddatacube_set_data_path(self):
        fixeddatacube = DataCube(golf_path)
        fixeddatacube.stratigraphy_from("eta", dz=0.1)
        with pytest.raises(AttributeError):
            fixeddatacube.data_path = "/trying/to/change/path.nc"

    def test_fixeddatacube_set_dataio(self):
        fixeddatacube = DataCube(golf_path)
        fixeddatacube.stratigraphy_from("eta", dz=0.1)
        with pytest.raises(AttributeError):
            fixeddatacube.dataio = 10  # io.NetCDF_IO(golf_path)

    def test_fixeddatacube_set_variables_list(self):
        fixeddatacube = DataCube(golf_path)
        fixeddatacube.stratigraphy_from("eta", dz=0.1)
        with pytest.raises(AttributeError):
            fixeddatacube.variables = ["is", "a", "list"]

    def test_fixeddatacube_set_variables_dict(self):
        fixeddatacube = DataCube(golf_path)
        fixeddatacube.stratigraphy_from("eta", dz=0.1)
        with pytest.raises(AttributeError):
            fixeddatacube.variables = {"is": True, "a": True, "dict": True}

    def test_fixeddatacube_set_planform_set_list(self):
        fixeddatacube = DataCube(golf_path)
        fixeddatacube.stratigraphy_from("eta", dz=0.1)
        with pytest.raises(AttributeError):
            fixeddatacube.planform_set = ["is", "a", "list"]

    def test_fixeddatacube_set_planform_set_dict(self):
        fixeddatacube = DataCube(golf_path)
        fixeddatacube.stratigraphy_from("eta", dz=0.1)
        with pytest.raises(AttributeError):
            fixeddatacube.planform_set = {"is": True, "a": True, "dict": True}

    def test_fixeddatacube_set_plans(self):
        fixeddatacube = DataCube(golf_path)
        fixeddatacube.stratigraphy_from("eta", dz=0.1)
        with pytest.raises(AttributeError):
            fixeddatacube.planforms = 10

    def test_fixeddatacube_set_section_set_list(self):
        fixeddatacube = DataCube(golf_path)
        fixeddatacube.stratigraphy_from("eta", dz=0.1)
        with pytest.raises(AttributeError):
            fixeddatacube.section_set = ["is", "a", "list"]

    def test_fixeddatacube_set_section_set_dict(self):
        fixeddatacube = DataCube(golf_path)
        fixeddatacube.stratigraphy_from("eta", dz=0.1)
        with pytest.raises(AttributeError):
            fixeddatacube.section_set = {"is": True, "a": True, "dict": True}

    def test_fixedset_set_sections(self):
        fixeddatacube = DataCube(golf_path)
        fixeddatacube.stratigraphy_from("eta", dz=0.1)
        with pytest.raises(AttributeError):
            fixeddatacube.sections = 10

    def test_export_frozen_variable(self):
        fixeddatacube = DataCube(golf_path)
        fixeddatacube.stratigraphy_from("eta", dz=0.1)
        frzn = fixeddatacube.export_frozen_variable("velocity")
        assert frzn.ndim == 3

    def test_section_with_stratigraphy(self):
        fixeddatacube = DataCube(golf_path)
        fixeddatacube.stratigraphy_from("eta", dz=0.1)
        assert hasattr(fixeddatacube, "strat_attr")
        sc = StrikeSection(fixeddatacube, distance_idx=10)
        assert sc.strat_attr is fixeddatacube.strat_attr
        _take = sc["velocity"][:, 1]
        assert _take.shape == (fixeddatacube.shape[0],)
        assert hasattr(sc, "strat_attr")
        _take2 = sc["velocity"].strat.as_preserved()
        assert _take2.shape == (
            fixeddatacube.shape[0],
            fixeddatacube.shape[2],
        )

    def test_register_variable(self):
        golf = DataCube(golf_path)
        golf.stratigraphy_from("eta", dz=0.1)
        golf.register_variable("testvar", np.zeros(golf.shape))
        assert "testvar" in golf.variables
        assert np.all(golf["testvar"].shape == golf.shape)


class TestStratigraphyCube:
    def test_no_tT_StratigraphyCube(self):
        # create a fixed cube for variable existing, type checks
        fixeddatacube = DataCube(golf_path)
        fixedstratigraphycube = StratigraphyCube.from_DataCube(fixeddatacube, dz=0.1)
        with pytest.raises(AttributeError):
            _ = fixedstratigraphycube.t
        with pytest.raises(AttributeError):
            _ = fixedstratigraphycube.T

    def test_export_frozen_variable(self):
        # create a fixed cube for variable existing, type checks
        fixeddatacube = DataCube(golf_path)
        fixedstratigraphycube = StratigraphyCube.from_DataCube(fixeddatacube, dz=0.1)
        frzn = fixedstratigraphycube.export_frozen_variable("time")
        assert frzn.ndim == 3

    def test_StratigraphyCube_inherit_varset(self):
        # create a fixed cube for variable existing, type checks
        fixeddatacube = DataCube(golf_path)
        # when creating from DataCube, varset should be inherited
        tempsc = StratigraphyCube.from_DataCube(fixeddatacube, dz=1)
        assert tempsc.varset is fixeddatacube.varset

    def test_auxiliary_data(self):
        fixeddatacube = DataCube(golf_path, auxdata="auxdata")
        fixedstratigraphycube = StratigraphyCube.from_DataCube(fixeddatacube, dz=0.1)
        assert fixedstratigraphycube.aux is fixeddatacube.aux

    def test_access_groups_manually(self):
        fixeddatacube = DataCube(golf_path, auxdata="auxdata")
        fixedstratigraphycube = StratigraphyCube.from_DataCube(fixeddatacube, dz=0.1)
        assert np.all(
            fixedstratigraphycube.dataio.dataset["auxdata"]["H_SL"]
            == fixedstratigraphycube.aux["H_SL"]
        )

    def test_set_aux_data(self):
        fixeddatacube = DataCube(golf_path)
        fixedstratigraphycube = StratigraphyCube.from_DataCube(fixeddatacube, dz=0.1)
        fixedstratigraphycube.set_aux("auxdata")
        assert np.all(
            fixedstratigraphycube.dataio.dataset["auxdata"]["H_SL"]
            == fixedstratigraphycube.aux["H_SL"]
        )

    def test_register_variable_self(self):
        fixeddatacube = DataCube(golf_path)
        fixedstratigraphycube = StratigraphyCube.from_DataCube(fixeddatacube, dz=0.1)
        fixedstratigraphycube.register_variable(
            "testvar", np.zeros(fixedstratigraphycube.shape)
        )
        assert "testvar" in fixedstratigraphycube.variables
        assert "testvar" not in fixeddatacube.variables
        assert np.all(
            fixedstratigraphycube["testvar"].shape == fixedstratigraphycube.shape
        )
        with pytest.raises(AttributeError):
            # parent cannot access var registered because would be wrong shape
            fixeddatacube["testvar"]
        with pytest.raises(ValueError):
            # try to register the wrong shape to the strat cube
            fixedstratigraphycube.register_variable(
                "testvar", np.zeros(fixeddatacube.shape)
            )

    def test_register_variable_parent(self):
        fixeddatacube = DataCube(golf_path)
        fixedstratigraphycube = StratigraphyCube.from_DataCube(fixeddatacube, dz=0.1)
        fixeddatacube.register_variable("testvar", np.zeros(fixeddatacube.shape))
        assert "testvar" in fixeddatacube.variables
        assert "testvar" in fixedstratigraphycube.variables
        # both can slice and both get "correct" shape
        assert np.all(fixeddatacube["testvar"].shape == fixeddatacube.shape)
        assert np.all(
            fixedstratigraphycube["testvar"].shape == fixedstratigraphycube.shape
        )

    def test_register_variable_bad_inputs(self):
        golf = DataCube(golf_path)
        with pytest.raises(ValueError, match=r"Input 'data' was incorrect"):
            golf.register_variable("testvar", np.zeros((10, 10, 10)))
        with pytest.raises(ValueError, match=r"Input 'data' was incorrect"):
            golf.register_variable("testvar", np.zeros((10, 10)))
        with pytest.raises(TypeError, match=r"Input 'name' was not"):
            golf.register_variable(33, "name")


class TestStratigraphyCubeSubsidence:
    def test_subsidence_cube(self):
        # create a cube with some uniform subsidence
        datacube = DataCube(golf_path)
        subsstratcube = StratigraphyCube.from_DataCube(
            datacube, dz=0.2, sigma_dist=0.005
        )
        nosubs = StratigraphyCube.from_DataCube(datacube, dz=0.2)

        assert subsstratcube.sigma_dist == 0.005
        assert nosubs.sigma_dist is None
        assert subsstratcube.sigma_dist != nosubs.sigma_dist
        assert nosubs.strata[0, -1, -1] == -2.0
        assert nosubs.strata[-1, -1, -1] == -2.0
        _expected_0 = -2.0 - (subsstratcube.sigma_dist * (datacube.shape[0] - 1))
        assert subsstratcube.strata[0, -1, -1] == pytest.approx(_expected_0)
        _expected_last = -2.0
        assert subsstratcube.strata[-1, -1, -1] == pytest.approx(_expected_last)


class TestFrozenStratigraphyCube:
    def test_types(self):
        fixeddatacube = DataCube(golf_path)
        fixedstratigraphycube = StratigraphyCube.from_DataCube(fixeddatacube, dz=0.1)
        frozenstratigraphycube = fixedstratigraphycube.export_frozen_variable("time")
        assert isinstance(frozenstratigraphycube, xr.core.dataarray.DataArray)

    def test_matches_underlying_data(self):
        fixeddatacube = DataCube(golf_path)
        fixedstratigraphycube = StratigraphyCube.from_DataCube(fixeddatacube, dz=0.1)
        frozenstratigraphycube = fixedstratigraphycube.export_frozen_variable("time")
        assert not (frozenstratigraphycube is fixedstratigraphycube)
        frzn_log = frozenstratigraphycube.values[
            ~np.isnan(frozenstratigraphycube.values)
        ]
        fixd_log = fixedstratigraphycube["time"].values[
            ~np.isnan(fixedstratigraphycube["time"].values)
        ]
        assert frzn_log.shape == fixd_log.shape
        assert np.all(fixd_log == frzn_log)


class TestLegacyPyDeltaRCMCubes:
    def test_init_cube_from_path_rcm8(self):
        with pytest.raises(RuntimeError):
            _ = _get_rcm8_path()

    def test_init_cube_rcm8(self):
        with pytest.raises(RuntimeError):
            _ = rcm8()


class TestCubesFromDictionary:
    def test_DataCube_one_dataset(self):
        fixeddatacube = DataCube(golf_path)
        eta_data = fixeddatacube["eta"][:, :, :]
        dict_cube = DataCube({"eta": eta_data})
        assert isinstance(dict_cube["eta"], xr.core.dataarray.DataArray)
        assert dict_cube.shape == fixeddatacube.shape
        assert np.all(dict_cube["eta"] == fixeddatacube["eta"][:, :, :])

    def test_DataCube_one_dataset_numpy(self):
        fixeddatacube = DataCube(golf_path)
        eta_data = np.array(fixeddatacube["eta"][:, :, :])
        dict_cube = DataCube({"eta": eta_data})
        # the return is always dataarray!
        assert isinstance(dict_cube["eta"], xr.core.dataarray.DataArray)
        assert dict_cube.shape == fixeddatacube.shape

    def test_DataCube_one_dataset_partial(self):
        fixeddatacube = DataCube(golf_path)
        eta_data = fixeddatacube["eta"][:30, :, :]
        dict_cube = DataCube({"eta": eta_data})
        assert np.all(dict_cube["eta"] == fixeddatacube["eta"][:30, :, :])

    def test_DataCube_two_dataset(self):
        fixeddatacube = DataCube(golf_path)
        eta_data = fixeddatacube["eta"][:, :, :]
        vel_data = fixeddatacube["velocity"][:, :, :]
        dict_cube = DataCube({"eta": eta_data, "velocity": vel_data})
        assert np.all(dict_cube["eta"] == fixeddatacube["eta"][:, :, :])
        assert np.all(dict_cube["velocity"] == fixeddatacube["velocity"][:, :, :])

    @pytest.mark.xfail(NotImplementedError, reason="not implemented", strict=True)
    def test_StratigraphyCube_from_etas(self):
        fixeddatacube = DataCube(golf_path)
        eta_data = fixeddatacube["eta"][:, :, :]
        _ = StratigraphyCube({"eta": eta_data})

    @pytest.mark.xfail(NotImplementedError, reason="not implemented", strict=True)
    def test_StratigraphyCube_from_etas_numpy(self):
        fixeddatacube = DataCube(golf_path)
        eta_data = fixeddatacube["eta"][:, :, :]
        _ = StratigraphyCube({"eta": np.array(eta_data)})

    def test_no_metadata_integrated(self):
        fixeddatacube = DataCube(golf_path)
        eta_data = fixeddatacube["eta"][:30, :, :]
        dict_cube = DataCube({"eta": eta_data})
        assert dict_cube.aux is None
        with pytest.warns(DeprecationWarning):
            assert dict_cube.meta is None  # to be deprecated

    @pytest.mark.parametrize(
        "order",
        [
            ("station_id", "temperature"),  # 1-D first
            ("temperature", "station_id"),  # 3-D first
        ],
    )
    def test_DataCube_shape_from_coords_not_order(self, order):
        """Shape must come from provided coords, not dict insertion order."""
        shape = (10, 50, 60)
        dims = {
            "t_ax": np.arange(shape[0]),
            "y_ax": np.arange(shape[1]),
            "x_ax": np.arange(shape[2]),
        }

        mapping = {
            "station_id": np.arange(100),  # extra 1-D var
            "temperature": np.random.rand(*shape),  # main 3-D var
        }
        # build dict preserving the requested insertion order
        data = {name: mapping[name] for name in order}

        cube = DataCube(data, dimensions=dims)

        # shape is derived from coords (dims), regardless of variable order
        assert cube.shape == (len(dims["t_ax"]), len(dims["y_ax"]), len(dims["x_ax"]))
        # dimension names also respect the provided order
        assert cube._dataio.dims == list(dims.keys())


class TestReadMetaFallbacks:
    class FakeIO:
        """Very small IO stub exposing only what _read_coords_dims_variables_from_dataio uses."""

        def __init__(
            self,
            t=4,
            y=5,
            x=6,
            *,
            mesh=False,
            bad_coord=False,
            var_name="temperature",
            dnames=("t", "y", "x"),
            # NEW knobs to drive scanner behavior:
            scan_vars=None,  # list of names to appear in known_variables (scan order)
            raise_on=(),  # names that should raise when __getitem__ is called
            three_d=None,  # names that should return a 3-D DataArray
            two_d=(),  # names that should return a 2-D (y,x) DataArray
        ):
            self._shape = (t, y, x)
            self._dnames = dnames
            self.var_name = var_name

            # Force the "no dims" path so the code scans known_variables:
            self.dims = []
            self.known_variables = (
                list(scan_vars) if scan_vars is not None else [var_name]
            )
            self.known_coords = list(dnames)

            self._raise_on = set(raise_on)
            self._three_d = set(three_d) if three_d is not None else {var_name}
            self._two_d = set(two_d)

            # Build coords (1-D default; 2-D mesh or bad 3-D if requested)
            t_ax = np.arange(t)
            y_ax = np.arange(y)
            x_ax = np.arange(x)

            if mesh:
                yy, xx = np.meshgrid(y_ax, x_ax, indexing="ij")  # (y,x)
                coord_y, coord_x = yy, xx
            elif bad_coord:
                coord_y, coord_x = (
                    np.zeros((y, x, 2)),
                    x_ax,
                )  # invalid 3-D coord to hit TypeError
            else:
                coord_y, coord_x = y_ax, x_ax

            self.dataset = {
                dnames[0]: t_ax,
                dnames[1]: coord_y,
                dnames[2]: coord_x,
            }

        def __getitem__(self, key):
            # Simulate read failures during the scan
            if key in self._raise_on:
                raise KeyError("simulated read error for testing")

            t, y, x = self._shape
            d0, d1, d2 = self._dnames

            # Return a real 3-D DataArray for any requested name in three_d
            if key in self._three_d:
                return xr.DataArray(np.zeros((t, y, x)), dims=(d0, d1, d2), name=key)

            # Return a 2-D array for names in two_d (so scanner never finds a 3-D var)
            if key in self._two_d:
                return xr.DataArray(np.zeros((y, x)), dims=(d1, d2), name=key)

            # Otherwise treat it like a coordinate lookup (1-D or 2-D/invalid
            #     as configured)
            return self.dataset[key]

    def _fresh_cube(self, t=4, y=5, x=6):
        """Create any DataCube instance; we'll overwrite its IO before reading meta."""
        # trivial in-memory cube that successfully constructs
        da = xr.DataArray(np.zeros((t, y, x)), dims=("t", "y", "x"))
        return DataCube({"eta": da})

    def test_no_dims_scans_for_3d_var_and_builds_coords(self):
        """
        When IO has no .dims, cube should scan for a 3-D var and use its dim
        names.
        """
        t, y, x = 4, 5, 6
        fake_io = self.FakeIO(t=t, y=y, x=x, mesh=False, bad_coord=False)
        cube = self._fresh_cube(t, y, x)

        # Swap in the stub IO and re-run metadata discovery
        cube._dataio = fake_io
        cube._read_coords_dims_variables_from_dataio()

        # Indices/coords should come from the stubbed 1-D arrays
        assert np.array_equal(cube._dim0_coords, np.arange(t))
        assert np.array_equal(cube._dim1_coords, np.arange(y))
        assert np.array_equal(cube._dim2_coords, np.arange(x))

    def test_2d_meshgrid_coords_collapsed_to_1d(self):
        """
        If y/x are provided as 2-D meshgrids, they are collapsed to their 1-D
        axes.
        """
        t, y, x = 3, 7, 8
        fake_io = self.FakeIO(t=t, y=y, x=x, mesh=True, bad_coord=False)
        cube = self._fresh_cube(t, y, x)

        cube._dataio = fake_io
        cube._read_coords_dims_variables_from_dataio()

        # Collapsed coords must match the original 1-D ranges that produced the mesh
        assert np.array_equal(cube._dim1_coords, np.arange(y))  # from [:, 0]
        assert np.array_equal(cube._dim2_coords, np.arange(x))  # from [0, :]

    def test_invalid_coord_ndim_raises_typeerror(self):
        """Non 1-D/2-D coordinate arrays should raise a clear TypeError."""
        fake_io = self.FakeIO(mesh=False, bad_coord=True)
        cube = self._fresh_cube()

        cube._dataio = fake_io
        with pytest.raises(
            TypeError,
            match=r"(?i)shape of coordinate array was not 1[-\s]?d or 2[-\s]?d",
        ):
            cube._read_coords_dims_variables_from_dataio()

    def test_scan_3d_var_handles_getitem_error(self):
        """Cover the `except: continue` branch while scanning known_variables."""
        t, y, x = 2, 3, 4
        cube = self._fresh_cube(t, y, x)
        fake_io = self.FakeIO(
            t=t,
            y=y,
            x=x,
            scan_vars=["bad", "temperature"],  # scan order
            raise_on={"bad"},  # first var raises
            three_d={"temperature"},  # second var is the 3-D one
        )
        cube._dataio = fake_io
        cube._read_coords_dims_variables_from_dataio()

        # Confirm we built coords successfully from the stub
        assert np.array_equal(cube._dim0_coords, np.arange(t))
        assert np.array_equal(cube._dim1_coords, np.arange(y))
        assert np.array_equal(cube._dim2_coords, np.arange(x))

    def test_scan_3d_var_raises_when_none_found(self):
        """Cover the final ValueError when the scan never encounters a true 3-D var."""
        t, y, x = 2, 3, 4
        cube = self._fresh_cube(t, y, x)
        fake_io = self.FakeIO(
            t=t,
            y=y,
            x=x,
            scan_vars=["a", "b"],  # two variables to scan
            three_d=set(),  # none are 3-D
            two_d={"a", "b"},  # both are 2-D, so scan never finds ndim==3
        )
        cube._dataio = fake_io
        with pytest.raises(ValueError, match=r"Could not infer 3-D dimensions"):
            cube._read_coords_dims_variables_from_dataio()


class TestLandsatCube:
    def test_init_cube_from_path_hdf5(self):
        # with pytest.warns(UserWarning, match=r"Group with.*"):
        hdfcube = DataCube(hdf_path)
        assert hdfcube._data_path == hdf_path
        assert hdfcube.dataio.io_type == "file"
        assert hdfcube._planform_set == {}
        assert hdfcube._section_set == {}
        assert type(hdfcube.varset) is VariableSet

    def test_read_Blue_intomemory(self):
        # with pytest.warns(UserWarning, match=r"Group with.*"):
        landsatcube = DataCube(hdf_path)
        assert landsatcube._dataio._in_memory_data == {}
        assert landsatcube.variables == ["Blue", "Green", "NIR", "Red"]
        assert len(landsatcube.variables) == 4

        landsatcube.read("Blue")
        assert len(landsatcube.dataio._in_memory_data) == 1

    def test_read_all_intomemory(self):
        # with pytest.warns(UserWarning, match=r"Group with.*"):
        landsatcube = DataCube(hdf_path)
        assert landsatcube.variables == ["Blue", "Green", "NIR", "Red"]
        assert len(landsatcube.variables) == 4

        landsatcube.read(True)
        assert len(landsatcube.dataio._in_memory_data) == 4

    def test_read_invalid(self):
        # with pytest.warns(UserWarning, match=r"Group with.*"):
        landsatcube = DataCube(hdf_path)
        with pytest.raises(TypeError):
            landsatcube.read(5)

    def test_get_coords(self):
        # with pytest.warns(UserWarning, match=r"Group with.*"):
        landsatcube = DataCube(hdf_path)
        assert landsatcube.coords == ["time", "x", "y"]
        assert landsatcube._coords == ["time", "x", "y"]
