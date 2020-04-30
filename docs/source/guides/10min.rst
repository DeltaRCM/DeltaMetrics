******************
10-minute tutorial
******************

learn DeltaMetrics in ten minutes!

.. testsetup:: *

   import deltametrics as dm

Connect to data
===============

In your application, you will want to connect to a your own dataset, but more on that later. 
For now, let's use a sample dataset that is distributed with DeltaMetrics.

.. doctest::
    
    >>> import deltametrics as dm

    >>> rcm8cube = dm.sample_data.cube.rcm8()
    >>> type(rcm8cube)
    <class 'deltametrics.cube.Cube'>

This creates an instance of a :obj:`~deltametrics.cube.Cube` object, which is the central office in many operations in using DeltaMetrics.
Creating the `rcm8cube` connects to a dataset, but does not read any of the data into memory, allowing for efficient computation on large datasets.

Inspect which variables are available in the `rcm8cube`.

.. doctest::

    >>> rcm8cube.variables
    ['x', 'y', 'time', 'eta', 'stage', 'depth', 'discharge', 'velocity', 'strata_age', 'strata_sand_frac', 'strata_depth']
    

Accessing data from Cube
========================

A :obj:`~deltametrics.cube.Cube` can be sliced directly by variable name.
Slicing a cube returns an instance of :obj:`~deltametrics.cube.CubeVariable`, which is a numpy ``ndarray`` compatible object; this means that it can be manipulated exactly as a standard ``ndarray``.

.. doctest::

    >>> type(rcm8cube['velocity'])
    <class 'deltametrics.cube.CubeVariable'>
    >>> type(rcm8cube['velocity'].base)
    <class 'numpy.ndarray'>


The data cube is most often interacted with by taking horizontal or vertical "cuts" of the cube. 
In this package, we refer to horizontal cuts as "plans" or (`Planform` data) and vertical cuts as "sections" (`Section` data). 

The :doc:`Planform <../reference/plan/index>` and :doc:`Section <../reference/section/index>` data types have a series of helpful classes and functions, which are fully documented in their respective pages.



Planform data
-------------

We can visualize Planform data of the cube with a number of built-in
functions. Let's inspect the state of several variables
of the Cube at the fortieth (40th) timestep:

.. doctest::

    >>> import matplotlib.pyplot as plt

    >>> fig, ax = plt.subplots(1, 3)
    >>> rcm8cube.show_plan('eta', t=40, ax=ax[0])
    >>> rcm8cube.show_plan('velocity', t=40, ax=ax[1], ticks=True)
    >>> rcm8cube.show_plan('strata_sand_frac', t=40, ax=ax[2])
    >>> plt.show()

.. plot:: guides/10min_three_plans.py


Section data
------------

For the sake of simplicity, this documentation uses the :obj:`~deltametrics.section.StrikeSection` as an example, but the following lexicon generalizes across the Section classes.

For a data cube, sections are most often instantiated by the :obj:`~deltametrics.cube.Cube.register_section` method:

.. doctest::

    >>> rcm8cube.register_section('demo', dm.section.StrikeSection(y=10))

which can then be accessed via the :obj:`~deltametrics.cube.Cube.sections` attribute of the Cube.

.. doctest::

    >>> rcm8cube.sections['demo']
    <deltametrics.section.StrikeSection object at 0x...>

Visualize all of the available sections as stratigraphy:

.. doctest::

    >>> fig, ax = plt.subplots(6, 1, sharex=True, figsize=(8,5))
    >>> ax = ax.flatten()
    >>> for i, var in enumerate(rcm8cube.dataio.known_variables):
    ...    rcm8cube.show_section('demo', var, ax=ax[i])
    >>> plt.show()

.. plot:: guides/10min_all_sections_strat.py
