import matplotlib.gridspec as gs

rcm8cube = dm.sample_data.rcm8()
rcm8cube.stratigraphy_from('eta')
rcm8cube.register_section('demo', dm.section.StrikeSection(y=10))

sc8cube = dm.cube.StratigraphyCube.from_DataCube(rcm8cube, dz=0.05)
sc8cube.register_section('demo', dm.section.StrikeSection(y=10))

fs = sc8cube.export_frozen_variable('strata_sand_frac')
fe = sc8cube.Z  # exported volume does not have coordinate information!

fig, ax = plt.subplots(figsize=(10, 2))
pcm = ax.pcolormesh(np.tile(np.arange(fs.shape[2]), (fs.shape[0], 1)),
                    fe[:, 10, :], fs[:, 10, :], shading='auto',
                    cmap=rcm8cube.varset['strata_sand_frac'].cmap,
                    vmin=rcm8cube.varset['strata_sand_frac'].vmin,
                    vmax=rcm8cube.varset['strata_sand_frac'].vmax)
dm.plot.append_colorbar(pcm, ax)
