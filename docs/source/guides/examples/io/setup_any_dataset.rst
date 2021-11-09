How to set up any dataset to work with DeltaMetrics
---------------------------------------------------

This guide describes how to set up any `t-x-y` dataset to work with DeltaMetrics. 


Connecting with NetCDF
~~~~~~~~~~~~~~~~~~~~~~

The standard I/O format for data used in DeltaMetrics is a NetCDF4 file, structured as sets of arrays. 
NetCDF was designed with dimensional data in mind, so that we can use common dimensions to 