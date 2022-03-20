View a section slice
--------------------

.. plot::
    :include-source:
    :context: close-figs

    >>> golfcube = dm.sample_data.golf()
    >>> golfstrat = dm.cube.StratigraphyCube.from_DataCube(
    ...     golfcube, dz=0.1)
    >>> circular = dm.section.CircularSection(golfstrat, radius=2000)

You can visualize the data yourself, or use the built-in `show()` method of a `Section`.

.. plot::
    :include-source:
    :context:

    >>> fig, ax = plt.subplots(1, 2, figsize=(7, 3))
    >>> ax[0].imshow(circular['velocity'])   # display directly
    >>> circular.show('velocity', ax=ax[1])  # use the built-in show()
    >>> plt.show()

.. hint::

    Use ``origin='lower'`` in `imshow` if you plan to show the data yourself!


.. seealso::

    :doc:`/guides/subject_guides/section`
        Subject guide on Section operations and classes

    :doc:`/guides/subject_guides/visualization`
        Subject guide on visualization in DeltaMetrics
