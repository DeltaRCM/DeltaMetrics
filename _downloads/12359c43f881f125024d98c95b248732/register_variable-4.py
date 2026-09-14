# preallocate an array and then populate with stratigraphic data
sqrt_new_var_spacetime = xr.full_like(test_data["eta"], np.nan)
sqrt_new_var_spacetime.data[
    test_strat.data_coords[:, 0],
    test_strat.data_coords[:, 1],
    test_strat.data_coords[:, 2],
] = test_strat.dataio["sqrt_new_var"].data[
    test_strat.strata_coords[:, 0],
    test_strat.strata_coords[:, 1],
    test_strat.strata_coords[:, 2],
]