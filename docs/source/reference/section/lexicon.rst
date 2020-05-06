
.. rubric:: Section lexicon

The `Section` module defines some terms that are used throughout the code and rest of the documentation. 

Most importantly, a Section is defined by a set of coordinates in the x-y plane of a cube.

Therefore, we transform variable definitions when extracting the section, and the coordinate system of the section is defined by the along-section direction :math:`s` and a vertical section coordinate, which is :math:`z` when viewing stratigraphy, and :math:`t` when viewing a spacetime section.

The data that make up the section can view the section as a `spacetime` section by simply calling the 

.. plot:: section/section_lexicon.py
    :include-source:

