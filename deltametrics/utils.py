import os, sys



def format_number(number):
    integer = int(round(number, -1))
    string = "{:,}".format(integer)
    return(string)



def format_table(number):
    integer = (round(number, 1))
    string = str(integer)
    return(string)



def _get_version():
    """
    Extract version number from single file, and make it availabe everywhere.
    """
    from . import _version
    return _version.__version__()
