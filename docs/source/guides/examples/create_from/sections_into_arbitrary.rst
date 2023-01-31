Creating Sections into different types of objects
-------------------------------------------------

Conventionally, we draw `Section` objects into a `Cube`. However, it is also possible to create `Section` objects into other objects, such as a `Mask` or `Planform` or any arbitrary array-like data. 

First, let's create a `Section` into a `Cube` and a basic `Planform` and compare.

.. plot::
    :include-source:
    :context: reset

    golfcube = dm.sample_data.golf()
    pl = dm.plan.Planform(golfcube, idx=-1)
    
    css = dm.section.StrikeSection(golfcube, distance=1200)
    pss = dm.section.StrikeSection(pl, distance=1200)
    
    fig, ax = plt.subplots()
    golfcube.quick_show('eta', idx=-1)
    css.show_trace(ax=ax)
    pss.show_trace('--', ax=ax)
    plt.show()


Because the `css` section is underlain by a `Cube`, it returns variables with dimensions `time x along section`. Whereas, the `pss` section is underlain by a `Planform` at time index `-1`, it returns variables with dimensions `along section`.

If we plot the elevation returned from `pss` and the last time index of the `css`, we can see that the underlying data are the same!

.. plot::
    :include-source:
    :context: close-figs

    fig, ax = plt.subplots()

    ax.plot(css.s, css['eta'][-1, :])
    ax.plot(pss.s, pss['eta'], '--')

    plt.show()


Similarly to creating a `Section` into the `Planform`, we can use an underlying `Mask`.

.. plot::
    :include-source:
    :context:
    
    EM = dm.mask.ElevationMask(
        golfcube["eta"][-1],
        elevation_threshold=0)
    mss = dm.section.StrikeSection(EM, distance=1200)

    ax.plot(mss.s, mss['mask'], ':')

    plt.show()


.. important::

    A common "gotcha" is forgetting that some `Section` types (e.g., :obj:`~deltametrics.section.CircularSection` and :obj:`~deltametrics.section.RadialSection`) will try to guess a useful `origin` during instantiation, by reading various attributes of an underlying `DataCube`. These attributes are not available during instantiation of a `Section` that reads into a `Planform` or `Mask`. This can lead to discrepencies in the locations of the Section objects!

    In this example, the `Section` into the `Planform` is offset by `L0`, because this attribute is not known to the `Planform`.
    
    .. code::

        ccs = dm.section.CircularSection(golfcube, radius=1000)
        pcs = dm.section.CircularSection(pl, radius=1000)

    .. plot::
        :context: close-figs

        ccs = dm.section.CircularSection(golfcube, radius=1000)
        pcs = dm.section.CircularSection(pl, radius=1000)
        
        fig, ax = plt.subplots()
        golfcube.quick_show('eta', idx=-1)
        ccs.show_trace(ax=ax)
        pcs.show_trace('--', ax=ax)
        plt.tight_layout()
        plt.show()


Arbitrary data
~~~~~~~~~~~~~~

You can also create a `Section` into any array-like data. 

.. plot::
    :include-source:
    :context: close-figs

    arr = np.random.uniform(size=(100, 200))
    arrss = dm.section.StrikeSection(arr, distance_idx=30)

.. note::

    There are no variable names associated with a single array, but you still need to specify an argument when slicing the section. You can use anything, but we suggest `[None]`.

    .. plot::
        :include-source:
        :context:

        fig, ax = plt.subplots()
        ax.plot(arrss[None])
        plt.show()
