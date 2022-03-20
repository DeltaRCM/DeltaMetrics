.. _quick-non-standard-testing:

Quick connect to your own  data
-------------------------------

If you are considering DeltaMetrics for your project, it may be helpful to use some of your data and see how DeltaMetrics works with that data, before committing to formatting a NetCDF file.

This guide covers how to connect to non-standard data with DeltaMetrics

.. note::

    The preferred approach to using DeltaMetrics is to :doc:`properly format a NetCDF file for input datasets <./setup_any_dataset>`.

First, use `numpy` to manipulate each variable in your dataset into a 3D array with "time" on the first axis, and the two spatial dimensions on the next two axes.
In the following example, the data is from a 100x200 domain with 51 saved data intervals.

.. plot::
    :include-source:
    :context: reset

    eta = np.random.uniform(0, 1, size=(51, 100, 200))

    dict_datacube = dm.cube.DataCube(
        {'eta': eta})

That is all you need to get your data into DeltaMetrics to test the package out.
You could also have additional variables by adding key-value pairs to the input dictionary.

.. plot::
    :include-source:
    :context:

    dict_datacube.quick_show('eta', idx=-1)

.. important::
    
    If you decide to use DeltaMetrics, please check out the guide on how to :doc:`properly format a NetCDF file for input datasets <./setup_any_dataset>`.
