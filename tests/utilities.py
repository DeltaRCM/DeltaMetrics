import sys
import os

import pytest
import netCDF4


def create_dummy_netcdf(tmp_path):
    """Create blank NetCDF4 file."""
    d = tmp_path / 'netcdf_files'
    try:
        d.mkdir()
    except Exception:
        pass
    p = d / 'dummy.nc'
    f = netCDF4.Dataset(p, "w", format="NETCDF4")
    f.createVariable('test', 'f4')
    f.close()
    return p


def create_dummy_txt_file(tmp_path):
    """Create a dummy text file."""
    d = tmp_path / 'txt_files'
    try:
        d.mkdir()
    except Exception:
        pass
    p = d / 'dummy.txt'
    _fobj = open(p, 'x')
    _fobj.close()
    return p
