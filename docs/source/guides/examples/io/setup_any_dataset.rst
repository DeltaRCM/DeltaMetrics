How to set up any dataset to work with DeltaMetrics
---------------------------------------------------

This guide describes how to set up any spatiotemporal (`t-x-y`) dataset to work with DeltaMetrics. 


Connecting with NetCDF
~~~~~~~~~~~~~~~~~~~~~~

The standard I/O format for data used in DeltaMetrics is a NetCDF4 file, structured as sets of arrays. 
NetCDF was designed with dimensional data in mind, so that we can use common dimensions to align and manipulate multiple different underlying variables.

.. important::

	Taking the time to set your data up correctly in a netCDF file is the preferred way to use  DeltaMetrics with your own data.

This guide is not meant to be an exhaustive guide on the netCDF format, so we provide only a simple overview of some core components that affect DeltaMetrics.

* `dataset`: the file, including all of the data and metadata to describe that data
* `variable`: a field in the `dataset` that contains numeric information
* `dimension`: a field in the `dataset` describing one dimension of the underlying data variables
* `group`: a mechanism for constructing a hierarchy of information within the `dataset`.

DeltaMetrics NetCDF data standards
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For DeltaMetrics to correctly work with underlying data, you must properly configure the dimensions and variables of the dataset.

* DeltaMetrics expects your core underlying spatiotemporal data (like 2D fields of elevation data, velocity, etc) to be organized into 3D arrays, with time along the first axis and two spatial dimensions along the next two axes.
* DeltaMetrics expects *at least* three `dimensions` defined in the `dataset`, one of which must be named ``time``, and the other two can have any name that describes the spatial dimensions of the data (e.g., `x`, `lon`, `easting`, etc.).
* DeltaMetrics expects a `variable` with name *exactly matching* the name of each of the three previous `dimensions`.
* DeltaMetrics expects some number of `variables` with arbitrary names that each contain a 3D array of spatiotemporal data of interest. I.e., this is the actual model/field/experiment data.
* DeltaMetrics expects there to be a `group` with name `meta`, which contains any information relevant to the spatiotemporal data. E.g., sea level, coordinates of sediment feed location.


Sample code for creating a DeltaMetrics NetCDF file with Python
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

First, let's make some sample data

.. plot::
	:include-source:
	:context: reset

	import numpy as np
	import matplotlib.pyplot as plt

	from netCDF4 import Dataset
	import os

	import deltametrics as dm

	## create the model data
	# some spatial information
	x = np.arange(-6, 6, 0.1)
	y = np.arange(-3, 3, 0.1)

	# some temporal information
	t = np.arange(1, 5)

	# meshes for each to make fake data
	T, Y, X = np.meshgrid(t, y, x, indexing='ij')

	# make fake time by x by y data
	#   hint: this would be the data from a model, experiment, or the field
	eta = np.sin(T * X + Y)
	velocity = np.cos(T * Y + X)
	H_SL = np.linspace(0.25, 0.9, num=len(t)) # sea level

	# check out the data before we save it into a netcdf file
	print("eta shape:", eta.shape)
	fig, ax = plt.subplots(2, len(t), figsize=(7, 3))
	for i, _t in enumerate(t):
	    ax[0, i].imshow(eta[i, :, :])
	    ax[1, i].imshow(velocity[i, :, :])
	plt.show()

Now, we write out the data to a netCDF file.

.. hint::

	You can use `None` as the length of the time dimension, if you want to create the NetCDF file while your model runs, and you do not know the size a priori.

.. plot::
	:context: close-figs

	import tempfile

	output_folder = tempfile.gettempdir()
	file_path = os.path.join(output_folder, 'fakemodel_output.nc')

.. code::

	output_folder = './output'
	file_path = os.path.join(output_folder, 'fakemodel_output.nc')

.. plot::
	:include-source:
	:context: close-figs

	## create and fill the netcdf file 
	output_netcdf = Dataset(file_path, 'w',
	                        format='NETCDF4')

	# add some description information (see netCDF docs for more)
	output_netcdf.description = 'Output from MyFakeModel'
	output_netcdf.source = 'MyFakeModel v0.1'

	# create time and spatial netCDF dimensions
	output_netcdf.createDimension('time', T.shape[0])  
	output_netcdf.createDimension('y', T.shape[1])
	output_netcdf.createDimension('x', T.shape[2])

	# create time and spatial netCDF variables
	v_time = output_netcdf.createVariable(
	    'time', 'f4', ('time',))
	v_time.units = 'second'
	v_x = output_netcdf.createVariable(
	    'x', 'f4', ('x'))
	v_x.units = 'meter'
	v_y = output_netcdf.createVariable(
	    'y', 'f4', ('y'))
	v_y.units = 'meter'

	# fill the variables with the coordinate information
	v_time[:] = t
	v_x[:] = x
	v_y[:] = y

	# set up variables for output data grids
	v_eta = output_netcdf.createVariable(
	    'eta', 'f4', ('time', 'y', 'x'))
	v_eta.units = 'meter'
	v_velocity = output_netcdf.createVariable(
	    'velocity', 'f4', ('time', 'y', 'x'))
	v_velocity.units = 'meter/second'
	v_eta[:] = eta
	v_velocity[:] = velocity

	# set up metadata group and populate variables
	output_netcdf.createGroup('meta')
	v_L0 = output_netcdf.createVariable(  # a scalar, the inlet length
	    'meta/L0', 'f4', ())  # no dims for scalar
	v_L0.units = 'cell'
	v_L0[:] = 5
	v_H_SL = output_netcdf.createVariable( # an array, the sea level
	    'meta/H_SL', 'f4', ('time',))  # only has time dimensions
	v_H_SL.units = 'meters'
	v_H_SL[:] = H_SL

	# close the netcdf file
	output_netcdf.close()


Now, let's load the NetCDF file with DeltaMetrics. Make a cube by pointing to the directory and file location.

.. plot::
	:include-source:
	:context: close-figs

	fake = dm.cube.DataCube(os.path.join(output_folder, 'fakemodel_output.nc'))

	fig, ax = plt.subplots(2, len(t), figsize=(7, 3))
	for i, _t in enumerate(t):
	    ax[0, i].imshow(fake['eta'][i, :, :])
	    ax[1, i].imshow(fake['velocity'][i, :, :])
	plt.show()

And just to show that the components of sea level and elevation have been connected, as they would be in the sample data throughout this documentation:

.. plot::
	:include-source:
	:context: close-figs

	fig, ax = plt.subplots()
	dm.plot.aerial_view(
	    fake['eta'][-1, :, :],
	    datum=fake.meta['H_SL'][-1],
	    ticks=True, ax=ax)
	plt.show()


Naming conventions for information inside a data file
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

While not strictly necessary, it may be helpful to adhere to a naming convention that DeltaMetrics uses internally to define some common attributes of sedimentary systems. 


Spatiotemporal variable naming conventions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* `eta`: name the bed elevation field `eta`. You can work around DeltaMetrics with a field with any other name to represent bed elevation, but the default expected name is `eta`.


Metadata variable naming conventions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
* `H_SL`
* `L0`

.. hint:: 

	None of these variables need to be defined; you can always manually pass them to DeltaMetrics constructors, but following the convention when creating your data file will save you many keystrokes later.