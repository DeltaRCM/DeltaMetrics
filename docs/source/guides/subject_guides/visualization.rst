Visualization Guide
=====================

This guide covers the full range of visualization routines available as part of sandplover, and explains how to create your own visualization routines that build on top of DeltaMetrics.

There are built-in visualization tools in sandplover, which are for the most part attached to the object that you would want to visualize itself.
For example, to visualize a section:

.. code::

    spl.section.RadialSection(golfcube, azimuth=70).show("velocity")


.. warning::

    The automatic detection and styling of variables by name was removed in
    v0.6.0. Future updates may provide similar functionality in an improved
    manner.



Data Visualization Methods
~~~~~~~~~~~~~~~~~~~~~~~~~~

Cube display methods
--------------------

todo...


Planform display methods
------------------------

todo...


Section display methods
-----------------------

`DataCube` with computed "quick" stratigraphy may be visualized a number of different ways.


.. doctest::

    >>> golfcube = spl.sample_data.golf()
    >>> golfcube.stratigraphy_from("eta")
    >>> golfcube.register_section("demo", spl.section.StrikeSection(distance_idx=10))
    >>> _v = "velocity"

    >>> fig, ax = plt.subplots(3, 2, sharex=True, figsize=(8, 6))
    >>> golfcube.sections["demo"].show(
    ...     _v, style="lines", data="spacetime", ax=ax[0, 0]
    ... )  # doctest: +SKIP
    >>> golfcube.sections["demo"].show(
    ...     _v, style="shaded", data="spacetime", ax=ax[0, 1]
    ... )  # doctest: +SKIP
    >>> golfcube.sections["demo"].show(
    ...     _v, style="lines", data="preserved", ax=ax[1, 0]
    ... )  # doctest: +SKIP
    >>> golfcube.sections["demo"].show(
    ...     _v, style="shaded", data="preserved", ax=ax[1, 1]
    ... )  # doctest: +SKIP
    >>> golfcube.sections["demo"].show(
    ...     _v, style="lines", data="stratigraphy", ax=ax[2, 0]
    ... )  # doctest: +SKIP
    >>> golfcube.sections["demo"].show(
    ...     _v, style="shaded", data="stratigraphy", ax=ax[2, 1]
    ... )  # doctest: +SKIP
    >>> plt.show(block=False)  # doctest: +SKIP

.. plot:: guides/visualization_datacube_section_display_style.py
    :include-source: false
