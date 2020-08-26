*******************
Visualization Guide
*******************

This guide covers the full range of visualization routines available as part of DeltaMetrics, and explains how to create your own visualization routines that build on top of DeltaMetrics.



Section display types
=====================

`DataCube` with computed "quick" stratigraphy may be visualized a number of different ways.


.. doctest::

    >>> rcm8cube = dm.sample_data.cube.rcm8()
    >>> rcm8cube.stratigraphy_from('eta')
    >>> rcm8cube.register_section('demo', dm.section.StrikeSection(y=10))
    >>> _v = 'velocity'

    >>> fig, ax = plt.subplots(3, 2, sharex=True, figsize=(8, 6))
    >>> rcm8cube.sections['demo'].show(_v, style='lines', 
    ...     data='spacetime', ax=ax[0,0]) #doctest +SKIP
    >>> rcm8cube.sections['demo'].show(_v, style='shaded',
    ...     data='spacetime', ax=ax[0,1]) #doctest +SKIP
    >>> rcm8cube.sections['demo'].show(_v, style='lines',
    ...     data='preserved', ax=ax[1,0]) #doctest +SKIP
    >>> rcm8cube.sections['demo'].show(_v, style='shaded',
    ...     data='preserved', ax=ax[1,1]) #doctest +SKIP
    >>> rcm8cube.sections['demo'].show(_v, style='lines',
    ...     data='stratigraphy', ax=ax[2,0]) #doctest +SKIP
    >>> rcm8cube.sections['demo'].show(_v, style='shaded',
    ...     data='stratigraphy', ax=ax[2,1]) #doctest +SKIP
    >>> plt.show(block=False) #doctest +SKIP

.. plot:: guides/visualization_datacube_section_display_style.py
