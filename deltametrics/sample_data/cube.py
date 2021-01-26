import warnings

from . import rcm8 as moved_rcm8

warnings.simplefilter("default")


def rcm8():
    """DEPRECATED rcm8 data cube location.

    The `rcm8` data cube has moved to `deltametrics.sample_data.rcm8()`.
    Access is maintained here for backwards compatability, but will be removed
    in a future release.
    """
    warnings.warn(
        "The `rcm8` data cube has moved to "
        "`deltametrics.sample_data.rcm8()`. "
        "Access is maintained here for backwards compatability, "
        "but will be removed in a future release.",
        DeprecationWarning,
    )
    return moved_rcm8()
