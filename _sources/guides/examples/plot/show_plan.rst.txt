View a planform
---------------

DataCube
^^^^^^^^

.. plot::
    :include-source:
    :context: close-figs

    >>> golfcube = dm.sample_data.golf()
    >>> final = dm.plan.Planform(golfcube, idx=-1)

You can visualize the data yourself, or use the built-in `show()` method of a `Planform`.

.. plot::
    :include-source:
    :context:

    >>> fig, ax = plt.subplots(1, 2, figsize=(7, 3))
    >>> ax[0].imshow(final['velocity'])   # display directly
    >>> final.show('velocity', ax=ax[1])  # use the built-in show()
    >>> plt.show()


StratigraphyCube
^^^^^^^^^^^^^^^^

.. plot::
    :include-source:
    :context: close-figs

    >>> golfstrat = dm.cube.StratigraphyCube.from_DataCube(
    ...     golfcube, dz=0.1)
    >>> minus1 = dm.plan.Planform(golfstrat, z=-1)

You can visualize the data yourself, or use the built-in `show()` method of a `Planform`.

.. plot::
    :include-source:
    :context:

    >>> fig, ax = plt.subplots(1, 2, figsize=(7, 3))
    >>> ax[0].imshow(minus1['velocity'])   # display directly
    >>> minus1.show('velocity', ax=ax[1])  # use the built-in show()
    >>> plt.show()



.. seealso::

    :doc:`/guides/subject_guides/planform`
        Subject guide on Planform operations and classes

    :doc:`/guides/subject_guides/visualization`
        Subject guide on visualization in DeltaMetrics