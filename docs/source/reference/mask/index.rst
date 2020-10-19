.. api.distribution:

*********************************
Masking operations
*********************************

The package makes available routines for masking out planview attributes of
the delta.

This class is defined in ``deltametrics.mask``.

.. note::

   When `Mask` objects are instantiated with single time-slice x-y planform
   data (i.e., `L x W` data), the returned mask is expanded along the first
   dimension to return an object with dimensions `1 x L x W`. This ensures that all `Mask` objects will have the same number of dimensions, but may lead to confusion if writing your own functions. 


Mask classes
====================

.. currentmodule:: deltametrics.mask

.. autosummary::
	:toctree: ../../_autosummary

	BaseMask
	ChannelMask
	WetMask
	LandMask
	ShorelineMask
	EdgeMask
	CenterlineMask
