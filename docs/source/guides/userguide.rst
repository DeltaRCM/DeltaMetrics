**********
User Guide
**********

This documentation provides a




Examples
########

We maintain a library of "workflow" examples, which show how to do common, interesting, or cool analysis with DeltaRCM.

.. toctree::
  :maxdepth: 2

  examples/index




Setting up your coding environment
##################################

.. testsetup:: *

    import deltametrics as dm
    import numpy as np
    import scipy as sp
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gs

All of the documentation in this package assumes that you have imported the DeltaMetrics package as ``dm``:

.. doctest::

    >>> import deltametrics as dm

Additionally, we frequently rely on the `numpy` package, and `matplotlib`. We will assume you have imported these packages by their common shorthand as well; if we import other packages, or other modules from `matplotlib`, these imports will be declared!

.. doctest::

    >>> import numpy as np
    >>> import matplotlib.pyplot as plt


Create and manipulate a "DataCube"
##################################

DeltaMetrics centers around the use of “Cubes” in DeltaMetrics language are the central office that connects all the different modules and workflows together.

.. doctest::

    >>> rcm8cube = dm.sample_data.rcm8()
    >>> rcm8cube
    <deltametrics.cube.DataCube object at 0x...>

Creating the ``rcm8cube`` connects to a dataset, but does not read any of the data into memory, allowing for efficient computation on large datasets. The type of the ``rcm8cube`` is ``DataCube``.

Inspect which variables are available in the ``rcm8cube``.

.. doctest::

    >>> rcm8cube.variables
    ['eta', 'stage', 'depth', 'discharge', 'velocity', 'strata_sand_frac']

We can access the underlying variables by name. The returned object are xarray-accessors with coordinates ``t-x-y``.
For example, access variables as:

.. doctest::

    >>> type(rcm8cube['eta'])
    <class 'deltametrics.cube.CubeVariable'>
    >>> rcm8cube['eta'].shape
    (51, 120, 240)

Let’s examine the timeseries of bed elevations by taking slices out of the ``'eta'`` variable, at various indicies (``t``) along the 0th dimension.

.. doctest::

    >>> nt = 5
    >>> ts = np.linspace(0, rcm8cube['eta'].shape[0]-1, num=nt, dtype=np.int)  # linearly interpolate ts

    >>> fig, ax = plt.subplots(1, nt, figsize=(12, 2))
    >>> for i, t in enumerate(ts):
    ...     ax[i].imshow(rcm8cube['eta'][t, :, :], vmin=-5, vmax=0.5) #doctest: +SKIP
    ...     ax[i].set_title('t = ' + str(t)) #doctest: +SKIP
    ...     ax[i].axes.get_xaxis().set_ticks([]) #doctest: +SKIP
    ...     ax[i].axes.get_yaxis().set_ticks([]) #doctest: +SKIP
    >>> ax[0].set_ylabel('y-direction') #doctest: +SKIP
    >>> ax[0].set_xlabel('x-direction') #doctest: +SKIP
    >>> plt.show() #doctest: +SKIP

.. plot:: guides/userguide_bed_timeseries.py

.. note::

    The 0th dimension of the cube is the *time* dimension, and the 1st and 2nd dimensions are the `y` and `x` dimensions of the model domain, respectively. The `x` dimension is the *cross-channel* dimension, Implementations using non-standard data should permute datasets to match this convention.

The CubeVariable supports arbitrary math (using `xarray` for fast computations via CubeVariable.data syntax).
For example:

.. doctest::

    >>> # compute the change in bed elevation between the last two intervals above
    >>> diff_time = rcm8cube['eta'][ts[-1], ...] - rcm8cube['eta'][ts[-2], ...]

    >>> fig, ax = plt.subplots(figsize=(5, 3))
    >>> im = ax.imshow(diff_time, cmap='RdBu', vmax=abs(diff_time).max(), vmin=-abs(diff_time).max())
    >>> cb = dm.plot.append_colorbar(im, ax)  # a convenience function
    >>> plt.show() #doctest: +SKIP

.. plot:: guides/userguide_bed_elevation_change.py


Manipulating Planform data
##########################

In addition to indexing directly, slices across the `time` dimension of the cube are referred to as "Planform" cuts.

TODO


Default Colors in DeltaMetrics
##############################

You may have noticed the beautiful colors above, and be wondering: "how are the colors set?"
We use a custom object (:obj:`~deltametrics.plot.VariableSet`) to define common plotting properties for all plots.
The `VariableSet` supports all kinds of other controls, such as custom colormaps for any variable, addition of new defined variables, fixed color limits, color normalizations, and more.
You can also use these attributes of the `VariableSet` in your own plotting routines.

See the :ref:`default colors in DeltaMetrics here <default_styling>` for more information.

Additionally, there are a :doc:`number of plotting routines <../reference/plot/index>` that are helpful in visualizations.


Manipulating Section data
#########################

We are often interested in not only the spatiotemporal changes in the planform of the delta, but we want to know what is preserved in the subsurface.
In DeltaMetrics, we refer to this preserved history as the "stratigraphy", and we provide a number of convenient routines for computing stratigraphy and analyzing the deposits.

Importantly, the stratigraphy (or i.e., which voxels are preserved) is not computed by default when a Cube instance is created.
We must directly tell the Cube instance to compute stratigraphy by specifying which variable contains the bed elevation history, because this history dictates preservation.

Mainly, the API works by registering a section of a specified type, and
assigning it a name (“demo” below). Registered sections are accessed via
the ``sections`` attribute of the cube:

For a data cube, sections are most easily instantiated by the :obj:`~deltametrics.cube.Cube.register_section` method:

.. doctest::

    >>> rcm8cube.register_section('demo', dm.section.StrikeSection(y=10))

which creates a section across a constant y-value ``==10``.
The path of any `Section` in the ``x-y`` plane can always be accessed via the ``.trace`` attribute.
We can plot the trace on top the the final bed elevation to see where the section will be located.

.. doctest::

    >>> fig, ax = plt.subplots()
    >>> rcm8cube.show_plan('eta', t=-1, ax=ax, ticks=True)
    >>> ax.plot(rcm8cube.sections['demo'].trace[:,0],
    ...         rcm8cube.sections['demo'].trace[:,1], 'r--') #doctest: +SKIP
    >>> plt.show() #doctest: +SKIP

.. plot:: guides/userguide_strikesection_location.py

Any registered section can then be accessed via the :obj:`~deltametrics.cube.Cube.sections` attribute of the Cube (returns a `dict`).

.. doctest::

    >>> rcm8cube.sections['demo']
    <deltametrics.section.StrikeSection object at 0x...>

Available section types are ``PathSection``, ``StrikeSection``,
``DipSection``, and ``RadialSection``.
Notably, `Sections` do not refer to any variable in particular, so `Sections`
are sliced themselves, similarly to the cube.

.. doctest::

    >>> rcm8cube.register_section('demo', dm.section.StrikeSection(y=10))
    >>> rcm8cube.sections['demo']['velocity']
    DataSectionVariable([[0., 0., 0., ..., 0., 0., 0.],
                         [0., 0., 0., ..., 0., 0., 0.],
                         [0., 0., 0., ..., 0., 0., 0.],
                         ...,
                         [0., 0., 0., ..., 0., 0., 0.],
                         [0., 0., 0., ..., 0., 0., 0.],
                         [0., 0., 0., ..., 0., 0., 0.]], dtype=float32)

We can visualize sections:

.. doctest::

    >>> fig, ax = plt.subplots(3, 1, sharex=True, figsize=(12,6))
    >>> rcm8cube.show_section('demo', 'eta', ax=ax[0])
    >>> rcm8cube.show_section('demo', 'velocity', ax=ax[1])
    >>> rcm8cube.show_section('demo', 'strata_sand_frac', ax=ax[2])
    >>> plt.show() #doctest: +SKIP

.. plot:: guides/userguide_three_spacetime_sections.py


You can also create a standalone section, which is not registered to the cube, but still supports slicing from the underlying dataset.

.. doctest::

    >>> sass = dm.section.StrikeSection(rcm8cube, y=10)
    >>> np.all(sass['velocity'] == rcm8cube.sections['demo']['velocity']) #doctest: +SKIP
    True

.. _userguide_quick_stratigraphy:

"Quick" stratigraphy
--------------------

We have implemented support for rapid stratigraphy computation for visualization, and preserved-time statistics.
These quick stratigraphy computations create a mesh of preserved elevations and fill this matrix with values sliced out of the ``t-x-y`` data.

Notably, the full "boxy" stratigraphy computation is also quite fast.
More on that below.
Compute the quick stratigraphy as:

.. doctest::

    >>> rcm8cube.stratigraphy_from('eta')

Now, the ``DataCube`` has knowledge of stratigraphy, which we can further use to visualize preservation within the spacetime, or visualize as an actual stratigraphic slice.

.. doctest::

    >>> rcm8cube.sections['demo']['velocity'].as_preserved()
    masked_DataSectionVariable(
      data=[[0.0, 0.0, 0.0, ..., 0.0, 0.0, 0.0],
            [--, --, --, ..., --, --, --],
            [--, --, --, ..., --, --, --],
            ...,
            [--, --, --, ..., --, --, --],
            [--, --, --, ..., --, --, --],
            [--, --, --, ..., --, --, --]],
      mask=[[False, False, False, ..., False, False, False],
            [ True,  True,  True, ...,  True,  True,  True],
            [ True,  True,  True, ...,  True,  True,  True],
            ...,
            [ True,  True,  True, ...,  True,  True,  True],
            [ True,  True,  True, ...,  True,  True,  True],
            [ True,  True,  True, ...,  True,  True,  True]],
      fill_value=1e+20,
      dtype=float32)

.. doctest::

    >>> fig, ax = plt.subplots(3, 1, sharex=True, figsize=(12, 8))
    >>> rcm8cube.show_section('demo', 'velocity', ax=ax[0])
    >>> rcm8cube.show_section('demo', 'velocity', data='preserved', ax=ax[1])
    >>> rcm8cube.show_section('demo', 'velocity', data='stratigraphy', ax=ax[2])
    >>> plt.show() #doctest: +SKIP

.. plot:: guides/userguide_quick_stratigraphy_sections.py


Quick stratigraphy makes it easy to visualize the behavior of the model across each of the variables:

.. doctest::

    >>> fig, ax = plt.subplots(7, 1, sharex=True, sharey=True, figsize=(12, 12))
    >>> ax = ax.flatten()
    >>> for i, var in enumerate(['time'] + rcm8cube.dataio.known_variables):
    ...     rcm8cube.show_section('demo', var, ax=ax[i], label=True,
    ...       style='shaded', data='stratigraphy')
    >>> plt.show() #doctest: +SKIP


.. plot:: guides/userguide_quick_stratigraphy_all_variables.py


All Section types
-----------------

There are multiple section types available.
The following are currently implemented.

.. doctest::

    >>> _strike = dm.section.StrikeSection(rcm8cube, y=18)
    >>> _path = dm.section.PathSection(rcm8cube, path=np.column_stack((np.linspace(50, 150, num=4000, dtype=np.int),
    ...                                                                np.linspace(10, 90, num=4000, dtype=np.int))))
    >>> _circ = dm.section.CircularSection(rcm8cube, radius=30)
    >>> _rad = dm.section.RadialSection(rcm8cube, azimuth=70)


The `Section` classes all inherit from the same ``BaseSection`` class, which means they mostly have the same options available to them, and have a common API.
Each has unique instantiation arguments, though, which must be properly specified.

.. doctest::

    >>> fig = plt.figure(constrained_layout=True, figsize=(10, 8))
    >>> spec = gs.GridSpec(ncols=2, nrows=3, figure=fig)
    >>> ax0 = fig.add_subplot(spec[0, :])
    >>> axs = [fig.add_subplot(spec[i, j]) for i, j in zip(np.repeat(np.arange(1, 3), 2), np.tile(np.arange(2), (3,)))]

    >>> rcm8cube.show_plan('eta', t=-1, ax=ax0, ticks=True)
    >>> for i, s in enumerate([_strike, _path, _circ, _rad]):
    ...     ax0.plot(s.trace[:,0], s.trace[:,1], 'r--') #doctest: +SKIP
    ...     s.show('velocity', ax=axs[i]) #doctest: +SKIP
    ...     axs[i].set_title(s.section_type) #doctest: +SKIP
    >>> plt.show() #doctest: +SKIP

.. plot:: guides/userguide_section_type_demos.py


Computing and Manipulating Stratigraphy
#######################################

:ref:`Quick stratigraphy <userguide_quick_stratigraphy>` works great for statistics of what-is-preserved and for quick visualizations, but it has several limitations.
1) Does not consider volume of sediment filled by preserved-time indicies, 2) cannot be sliced by planform, 3) irregularity does not lend well to computation and other uses (hydrological studies).

So, we want to be able to create what I refer to as "boxy" stratigraphy.
This has been done in the past by "placing" values from, e.g., ``strata_sand_frac`` into stratigraphy.
This requires full computation for any variable you want to examine though.
Here, we use a method that computes boxy stratigraphy only once, then synthesizes the volume from
the precomputed sparse indicies.

Here’s a simple example to demonstrate how we place data into the stratigraphy.

.. doctest::

    >>> ets = rcm8cube['eta'][:, 25, 120]  # a "real" slice of the model
    >>> fig, ax = plt.subplots(figsize=(8, 4))
    >>> dm.plot.show_one_dimensional_trajectory_to_strata(ets, ax=ax, dz=0.25)
    >>> plt.show() #doctest: +SKIP

.. plot:: guides/userguide_1d_example.py


Begin by creating a ``StratigraphyCube``:

.. doctest::

    >>> sc8cube = dm.cube.StratigraphyCube.from_DataCube(rcm8cube, dz=0.05)
    >>> sc8cube.variables
    ['eta', 'stage', 'depth', 'discharge', 'velocity', 'strata_sand_frac']


We can then slice this cube in the same way as the ``DataCube``, but what we get back is *stratigraphy* rather than *spacetime*.
Compare the slice from the `rcm8cube` (left) to the `sc8cube` (right):

.. doctest::

    >>> fig, ax = plt.subplots(1, 2, figsize=(8, 2))
    >>> rcm8cube.sections['demo'].show('velocity', ax=ax[0]) #doctest: +SKIP
    >>> sc8cube.sections['demo'].show('velocity', ax=ax[1]) #doctest: +SKIP
    >>> plt.show() #doctest: +SKIP

.. plot:: guides/userguide_compare_slices.py


Validation of the stratigraphy is easily seen by looking at the ``time`` attribute.
Note that sections are *not* inherited from the ``DataCube`` by default (we’re working on this and related features).

Let’s add a section at the same location as ``rcm8cube.sections['demo']``.

.. doctest::

    >>> sc8cube.register_section('demo', dm.section.StrikeSection(y=10))
    >>> sc8cube.sections
    {'demo': <deltametrics.section.StrikeSection object at 0x...>}

Let's examine the stratigraphy in three different visual styles.

.. doctest::

    >>> fig, ax = plt.subplots(3, 1, sharex=True, sharey=True, figsize=(12, 8))
    >>> rcm8cube.sections['demo'].show('time', style='lines', data='stratigraphy', ax=ax[0], label=True)
    >>> sc8cube.sections['demo'].show('time', ax=ax[1])
    >>> rcm8cube.sections['demo'].show('time', data='stratigraphy', ax=ax[2])
    >>> plt.show() #doctest: +SKIP

.. plot:: guides/userguide_three_stratigraphy.py

Similar to the demonstration above, each variable (property) of the underlying cube can be displayed. These displays utilize the same *precomputed* locations in the stratigraphy and simply filled the synthesized matrix with the different variable values.

.. doctest::

    >>> fig, ax = plt.subplots(7, 1, sharex=True, sharey=True, figsize=(12, 12))
    >>> ax = ax.flatten()
    >>> for i, var in enumerate(['time'] + sc8cube.dataio.known_variables):
    ...     sc8cube.show_section('demo', var, ax=ax[i], label=True,
    ...                          style='shaded', data='stratigraphy')
    >>> plt.show() #doctest: +SKIP

.. plot:: guides/userguide_all_vars_stratigraphy.py

The stratigraphy cube allows us to slice planform slabs of stratigraphy
too. We are working on a method to more easily slice by elevation
values. This might be done by subclassing ``xarray`` rather than
``numpy`` for basic data arrays.

.. doctest::

    >>> elev_idx = (np.abs(sc8cube.z - -2)).argmin()  # find nearest idx to -2 m

    >>> fig, ax = plt.subplots(figsize=(5, 3))
    >>> sc8cube.show_plan('strata_sand_frac', elev_idx, ticks=True)
    >>> plt.show() #doctest: +SKIP

.. plot:: guides/userguide_stratigraphy_planform_slice.py


Frozen stratigraphy volumes
---------------------------

We still support creating “frozen” cubes, which might be useful for to
speed up computations if an array is being accessed over and over.

.. code::

    fs = sc8cube.export_frozen_variable('strata_sand_frac')
    fe = sc8cube.Z  # exported volume does not have coordinate information!

    fig, ax = plt.subplots(figsize=(10, 2))
    pcm = ax.pcolormesh(np.tile(np.arange(fs.shape[2]), (fs.shape[0], 1)),
       fe[:,10,:], fs[:,10,:], shading='auto',
       cmap=rcm8cube.varset['strata_sand_frac'].cmap,
       vmin=rcm8cube.varset['strata_sand_frac'].vmin,
       vmax=rcm8cube.varset['strata_sand_frac'].vmax)
    dm.plot.append_colorbar(pcm, ax)
    plt.show() #doctest: +SKIP

Note than you can also bypass the creation of a ``StratigraphyCube``,
and just directly obtain a frozen volume with:

.. doctest::

   >>> fs, fe = dm.strat.compute_boxy_stratigraphy_volume(rcm8cube['eta'], rcm8cube['strata_sand_frac'], dz=0.05)

However, this will require recomputing the stratigraphy preservation to create another cube in the future, and because the ``StratigraphyCube`` stores data on disk, the memory footprint is relatively small, and so we recommend just computing the ``StratigraphyCube`` and using the ``export_frozen_variable)`` method.
Finally, ``DataCubeVariable`` and ``StratigraphyCubeVariable`` support a ``.as_frozen()`` method themselves.

We should verify that the frozen cubes actually match the underlying data!

.. doctest::

    >>> np.all( fs[~np.isnan(fs)] == sc8cube['strata_sand_frac'][~np.isnan(sc8cube['strata_sand_frac'])] ) #doctest: +SKIP
    True

The access speed of a frozen volume is **much** faster than a live cube.
This is because the live cube does not store any data in memory.
Keeping data on disk is advantageous for large datasets, but slows down access considerably for computation.
**The speed of access in a frozen cube may be several thousand times faster, so it can be advantageous to export frozen cubes before computation.**
See a :doc:`demonstration of the speed comparison in the Examples library <examples/computations/comparing_speeds_of_stratigraphy_access>`.




Masks
#####

We have implemented operations to compute masks of several types. These
operations will be wrapped into the ``Plan`` API, and we will have
methods to create new “variables” in the data cube which hold the binary
values of the masks.

Currently implemented `Masks`:
  * ChannelMask
  * EdgeMask
  * LandMask
  * ShorelineMask
  * WetMask
  * CenterlineMask

.. doctest::

    >>> # use a new cube
    >>> maskcube = dm.sample_data.rcm8()

    >>> # create the masks from variables in the cube
    >>> land_mask = dm.mask.LandMask(maskcube['eta'][-1, :, :])
    >>> wet_mask = dm.mask.WetMask(maskcube['eta'][-1, :, :])
    >>> channel_mask = dm.mask.ChannelMask(maskcube['velocity'][-1, :, :], maskcube['eta'][-1, :, :])
    >>> centerline_mask = dm.mask.CenterlineMask(channel_mask)
    >>> edge_mask = dm.mask.EdgeMask(maskcube['eta'][-1, :, :])
    >>> shore_mask = dm.mask.ShorelineMask(maskcube['eta'][-1, :, :])

.. doctest::

    >>> fig = plt.figure(constrained_layout=True, figsize=(12, 10))
    >>> spec = gs.GridSpec(ncols=2, nrows=4, figure=fig)
    >>> ax0 = fig.add_subplot(spec[0, :])
    >>> axs = [fig.add_subplot(spec[i, j]) for i, j in zip(np.repeat(np.arange(1, 4), 2), np.tile(np.arange(2), (4,)))]

    >>> ax0.imshow(maskcube['eta'][-1, :, :]) #doctest: +SKIP
    >>> for i, m in enumerate([land_mask, wet_mask, channel_mask, centerline_mask, edge_mask, shore_mask]):
    ...     axs[i].imshow(m.mask[-1, :, :], cmap='gray') #doctest: +SKIP
    ...     axs[i].set_title(m.mask_type) #doctest: +SKIP
    >>> plt.show() #doctest: +SKIP

.. plot:: guides/userguide_masks_all_demo.py
