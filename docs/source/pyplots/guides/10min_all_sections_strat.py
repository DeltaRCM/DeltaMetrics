import matplotlib.pyplot as plt

import deltametrics as dm

rcm8cube = dm.sample_data.cube.rcm8()
rcm8cube.register_section('demo', dm.section.StrikeSection(y=10))

fig, ax = plt.subplots(6, 1, sharex=True, figsize=(8, 5))
ax = ax.flatten()
for i, var in enumerate(rcm8cube.dataio.known_variables):
    rcm8cube.show_section('demo', var, ax=ax[i])
plt.show()
