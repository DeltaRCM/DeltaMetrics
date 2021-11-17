
.. rubric:: Section lexicon

The `Section` module defines some terms that are used throughout the code and rest of the documentation. 

Most importantly, a Section is defined by a set of coordinates in the x-y plane of a `Cube`.

Therefore, we transform variable definitions when extracting the `Section`, and the coordinate system of the section is defined by the along-section direction :math:`s` and a vertical section coordinate, which is :math:`z` when viewing stratigraphy, and :math:`t` when viewing a spacetime section.

The data that make up the section can view the section as a `spacetime` section by simply calling a variable from the a section into a `DataCube`.

.. doctest::

    >>> rcm8cube = dm.sample_data.golf()
    >>> strike = dm.section.StrikeSection(rcm8cube, y=10)
    >>> strike['velocity']
    <xarray.DataArray 'velocity' (time: 101, s: 200)>
    array([[0.2   , 0.2   , 0.2   , ..., 0.2   , 0.2   , 0.2   ],
           [0.    , 0.    , 0.    , ..., 0.    , 0.    , 0.    ],
           [0.    , 0.0025, 0.    , ..., 0.    , 0.    , 0.    ],
           ...,
           [0.    , 0.    , 0.    , ..., 0.0025, 0.    , 0.    ],
           [0.    , 0.    , 0.    , ..., 0.    , 0.    , 0.    ],
           [0.    , 0.    , 0.    , ..., 0.0025, 0.    , 0.    ]],
          dtype=float32)
    Coordinates:
      * s        (s) float64 0.0 50.0 100.0 150.0 ... 9.85e+03 9.9e+03 9.95e+03
      * time     (time) float32 0.0 5e+05 1e+06 1.5e+06 ... 4.9e+07 4.95e+07 5e+07
    Attributes:
        slicetype:           data_section
        knows_stratigraphy:  False
        knows_spacetime:     True


If a `DataCube` has preservation information (i.e., if the :meth:`~deltametrics.cube.DataCube.stratigraphy_from()` method has been called), then the `xarray` object that is returned has this information too.
The same `spacetime` data can be requested in the "preserved" form, where non-preserved t-x-y points are masked with ``np.nan``.

.. doctest::

    >>> rcm8cube.stratigraphy_from('eta')
    >>> strike['velocity'].strat.as_preserved()
    <xarray.DataArray 'velocity' (time: 101, s: 200)>
    array([[0.2, 0.2, 0.2, ..., 0.2, 0.2, 0.2],
           [nan, nan, nan, ..., nan, nan, nan],
           [nan, nan, nan, ..., nan, nan, nan],
           ...,
           [nan, nan, nan, ..., nan, nan, nan],
           [nan, nan, nan, ..., nan, nan, nan],
           [nan, nan, nan, ..., nan, nan, nan]], dtype=float32)
    Coordinates:
      * s        (s) float64 0.0 50.0 100.0 150.0 ... 9.85e+03 9.9e+03 9.95e+03
      * time     (time) float32 0.0 5e+05 1e+06 1.5e+06 ... 4.9e+07 4.95e+07 5e+07
    Attributes:
        slicetype:           data_section
        knows_stratigraphy:  True
        knows_spacetime:     True

.. note::
    The section has access to the preservation information of the data, even though it was instantiated prior to the computation of preservation!


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
