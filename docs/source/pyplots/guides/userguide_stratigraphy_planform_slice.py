rcm8cube = dm.sample_data.cube.rcm8()

sc8cube = dm.cube.StratigraphyCube.from_DataCube(rcm8cube, dz=0.05)
sc8cube.register_section('demo', dm.section.StrikeSection(y=10))

elev_idx = (np.abs(sc8cube.z - -2)).argmin()  # find nearest idx to -2 m

fig, ax = plt.subplots(figsize=(5, 3))
sc8cube.show_plan('strata_sand_frac', elev_idx, ticks=True)
