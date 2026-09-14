import numpy as np
import matplotlib.pyplot as plt
import xarray as xr

import sandplover as spl

# Load sample data and instantiate Cubes
test_data = spl.sample_data.golf()  # DataCube
test_strat = spl.cube.StratigraphyCube.from_DataCube(test_data, dz=0.1)  # StratigraphyCube

# Create a synthetic 3D variable
new_variable = xr.zeros_like(test_data["eta"])
for t in np.arange(test_data.shape[0]):
    new_variable[t] = np.mod(
        np.sin(np.asarray(test_data["eta"][t]) * 12345.6789) * 43758.5453, 1
    )

# Register the new variable to the DataCube
test_data.register_variable("new_var", new_variable)