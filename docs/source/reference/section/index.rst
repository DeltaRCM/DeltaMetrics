.. api.profile:

*********************************
Section operations and classes
*********************************

The package makes available models for vertical sections of deposits and strata.

All classes inherit from :obj:`BaseSection`, and redefine methods and attributes for their own types.

The classes are defined in ``deltametrics.section``. 


.. include:: lexicon.rst


Section types
==============

.. currentmodule:: deltametrics.section

.. autosummary:: 
    :toctree: ../../_autosummary
    
    BaseSection
    StrikeSection
    DipSection
    RadialSection


Section returns
===============

.. autosummary:: 
    :toctree: ../../_autosummary

    BaseSectionVariable
    DataSectionVariable
    StratigraphySectionVariable

