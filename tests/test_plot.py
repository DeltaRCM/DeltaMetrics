import pytest

import sys
import os

import numpy as np
import matplotlib.pyplot as plt

from deltametrics import plot

rcm8_path = os.path.join(os.path.dirname(__file__), '..', 'deltametrics',
                         'sample_data', 'files', 'pyDeltaRCM_Output_8.nc')


def test_initialize_default_VariableInfo():
    vi = plot.VariableInfo('testinfo')
    assert vi.cmap.N == 64


@pytest.mark.xfail(raises=TypeError, strict=True)
def test_initialize_default_VariableInfo_noname():   
    vi = plot.VariableInfo()


@pytest.mark.xfail(raises=TypeError, strict=True)
def test_initialize_default_VariableInfo_name_isstr():   
    vi = plot.VariableInfo(None)


def test_initialize_VariableInfo_cmap_str():
    vi = plot.VariableInfo('testinfo', cmap='Blues')
    assert vi.cmap.N == 64
    assert vi.cmap(0)[0] == pytest.approx(0.96862745)


def test_initialize_VariableInfo_cmap_spec():
    vi = plot.VariableInfo('testinfo', cmap=plt.cm.get_cmap('Blues', 7))
    assert vi.cmap.N == 7
    assert vi.cmap(0)[0] == pytest.approx(0.96862745)


def test_initialize_VariableInfo_cmap_tuple():
    vi = plot.VariableInfo('testinfo', cmap=('Blues', 7))
    assert vi.cmap.N == 7
    assert vi.cmap(0)[0] == pytest.approx(0.96862745)


def test_initialize_VariableInfo_label_str():
    vi = plot.VariableInfo('testinfo', label='Test Information')
    assert vi.label == 'Test Information'
    assert vi.name == 'testinfo'


def test_VariableInfo_change_label():
    vi = plot.VariableInfo('testinfo')
    vi.label = 'Test Information'
    assert vi.label == 'Test Information'
    assert vi.name == 'testinfo'


def test_initialize_default_VariableSet():
    vs = plot.VariableSet()
    assert 'eta' in vs.known_list
    assert vs['depth'].vmin == 0


def test_initialize_VariableSet_override_known_VariableInfo():
    vi = plot.VariableInfo('depth')
    od = {'depth': vi}
    vs = plot.VariableSet(override_dict=od)
    assert vs['depth'].vmin is None


def test_initialize_VariableSet_override_unknown_VariableInfo():
    vi = plot.VariableInfo('fakevariable', vmin=-9999)
    od = {'fakevariable': vi}
    vs = plot.VariableSet(override_dict=od)
    assert vs['fakevariable'].vmin == -9999


@pytest.mark.xfail(raises=TypeError, strict=True)
def test_initialize_VariableSet_override_known_badtype():
    vi = plot.VariableInfo('depth')
    od = ('depth', vi)
    vs = plot.VariableSet(override_dict=od)


def test_VariableSet_add_known_VariableInfo():
    vs = plot.VariableSet()
    vi = plot.VariableInfo('depth', vmin=-9999)
    vs.depth = vi
    assert vs.depth.vmin == -9999


def test_VariableSet_add_unknown_VariableInfo():
    vs = plot.VariableSet()
    vi = plot.VariableInfo('fakevariable', vmin=-9999)
    vs.fakevariable = vi
    assert vs.fakevariable.vmin == -9999


def test_VariableSet_set_known_VariableInfo_direct():
    vs = plot.VariableSet()
    vs.depth.vmin = -9999
    assert vs.depth.vmin == -9999


def test_VariableSet_change_then_default():
    vs = plot.VariableSet()
    _first = vs.depth.cmap(0)[0]
    vi = plot.VariableInfo('depth', vmin=-9999)
    vs.depth = vi
    assert vs.depth.vmin == -9999
    vs.depth = None  # reset to default
    assert vs.depth.cmap(0)[0] == _first
    assert vs.depth.vmin == 0


@pytest.mark.xfail(raises=TypeError, strict=True)
def test_VariableSet_add_known_badtype():
    vs = plot.VariableSet()
    vs.depth = 'Yellow!'


@pytest.mark.xfail(raises=TypeError, strict=True)
def test_VariableSet_add_unknown_badtype():
    vs = plot.VariableSet()
    vs.fakevariable = 'Yellow!'
