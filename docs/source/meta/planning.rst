Planning
########

A working list of development goals will be useful when initially developing the package.

Mission statement
-----------------

The goal of the DeltaMetrics project is to provide convenient tools to analyze seidmentologic data cubes. 


Objectives
----------

#. Users should be able to instantiate the Cube object with a numpy array where dimensions are expected to be ``t-x-y``, and a value specifying the type of data stored in the array. Data could be, lidar scans, overhead photos, grain size maps (DeltaRCM), flow velocity (DeltaRCM), etc. An optional time array can be supplied, specifying time at each z; we can also accept a single value ``T``, and time is linearly spaced wrt ``T`` and ``0``.
#. Users should be able to extract ``Section`` s (e.g., :class:`~deltametrics.section.RadialSection`) from the ``Cube``. Several types of sectioning should be supported. By default, we should try to process to "stratigraphy" via an elevation timeseries attribute of the ``Cube``, but can have a flag to leave as raw section.
#. Users should be able to do processed and compute statistics on the ``Section`` s. For example, idenfying channels (?) or computing compensation statistics (extract a ``Horizon`` first?). These routines can live in ``deltametrics.strat`` (:doc:`../reference/strat/index`) or in other modules.
#. Users should be able to extract ``Plan`` s from the ``Cube``, which are a) single time horizon or b) single elevation horizon planviews of whatever attributes the user wants. 
#. Users should be able to compute statistics on these ``Plan`` s. For example, find shoreline, radially average delta size, extract channels, shoreline curvature, etc. These routines can live in ``deltametrics.plan`` (:doc:`../reference/plan/index`) or in other modules, such as ``.shoreline.py`` etc, or even in separate sub-packages as we deem appropriate. It may be helpful to separate out the extraction of water-land masks to a new module (:obj:`~deltametrics.mask`).
   

To this end, the ``Cube``, ``Section``, and ``Plan`` classes are the core classes of the package, and everything should be built around them.
