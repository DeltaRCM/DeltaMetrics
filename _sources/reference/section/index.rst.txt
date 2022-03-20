.. api.profile:

*********************************
Section operations
*********************************

The package makes available models for vertical sections of deposits and strata.

All classes inherit from :obj:`BaseSection`, and redefine methods and attributes for their own types.

The classes are defined in ``deltametrics.section``. 

.. hint::

  There is a complete :doc:`Sections Subject Guide </guides/subject_guides/mobility>` about the organization of this area of DeltaMetrics and examples for how to use and compute `Section` metrics.

Section classes
===============

.. currentmodule:: deltametrics.section

.. autosummary:: 
    :toctree: ../../_autosummary
    
    PathSection
    StrikeSection
    DipSection
    CircularSection
    RadialSection
    BaseSection
