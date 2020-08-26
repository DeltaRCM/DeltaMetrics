
.. rubric:: Section lexicon

The `Section` module defines some terms that are used throughout the code and rest of the documentation. 

Most importantly, a Section is defined by a set of coordinates in the x-y plane of a `Cube`.

Therefore, we transform variable definitions when extracting the `Section`, and the coordinate system of the section is defined by the along-section direction :math:`s` and a vertical section coordinate, which is :math:`z` when viewing stratigraphy, and :math:`t` when viewing a spacetime section.

The data that make up the section can view the section as a `spacetime` section by simply calling a `DataSectionVariable` directly.

.. doctest::

    >>> rcm8cube = dm.sample_data.cube.rcm8()
    >>> strike = dm.section.StrikeSection(rcm8cube, y=10)
    >>> strike['velocity']
    DataSectionVariable([[0., 0., 0., ..., 0., 0., 0.],
                         [0., 0., 0., ..., 0., 0., 0.],
                         [0., 0., 0., ..., 0., 0., 0.],
                         ...,
                         [0., 0., 0., ..., 0., 0., 0.],
                         [0., 0., 0., ..., 0., 0., 0.],
                         [0., 0., 0., ..., 0., 0., 0.]], dtype=float32)

If a `DataSectionVariable` has preservation information (i.e., if the :meth:`~deltametrics.cube.DataCube.stratigraphy_from()` method has been called), then the `DataSectionVariable`, may be requested in the "preserved" form, where non-preserved t-x-y cells are masked with ``np.nan``. Note that the section has access to the preservation information of the data, even though it was instantiated prior to the computation of preservation!

.. doctest::

    >>> rcm8cube.stratigraphy_from('eta')
    >>> strike['velocity'].as_preserved()
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

We can display the arrays using `matplotlib` to examine the spatiotemporal change of any variable; show the `velocity` in the below examples.

.. code::

    >>> fig, ax = plt.subplots(2, 1, sharex=True, figsize=(6, 3.5))
    >>> ax[0].imshow(rcm8cube.sections['demo']['velocity'],
    ...              origin='lower', cmap=rcm8cube.varset['velocity'].cmap)
    >>> ax[0].set_ylabel('$t$ coordinate')
    >>> ax[1].imshow(rcm8cube.sections['demo']['velocity'].as_preserved(),
    ...              origin='lower', cmap=rcm8cube.varset['velocity'].cmap)
    >>> ax[1].set_ylabel('$t$ coordinate')

.. plot:: section/section_lexicon.py

Note that in this visual all non-preserved spacetime points have been masked and are shown as white.
See the `numpy MaskedArray documentation <https://numpy.org/doc/stable/reference/maskedarray.generic.html>`_ for more information on interacting with masked arrays.
