#! /usr/bin/env python
from setuptools import setup, find_packages

from deltametrics import utils

setup(name='DeltaMetrics',
      version=utils._get_version(),
      author='The DeltaRCM Team',
      license='MIT',
      description="Tools for manipulating sedimentologic data cubes.",
      long_description=open('README.rst').read(),
      packages=find_packages(exclude=['*.tests']),
      include_package_data=True,
      url='https://github.com/DeltaRCM/DeltaMetrics',
      install_requires=['matplotlib', 'netCDF4',
                        'scipy', 'numpy', 'pyyaml', 'xarray'],
      )
