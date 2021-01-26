import matplotlib.pyplot as plt

import deltametrics as dm

rcm8cube = dm.sample_data.rcm8()
rcm8cube.stratigraphy_from('eta')
rcm8cube.register_section('demo', dm.section.StrikeSection(y=5))

fig, ax = plt.subplots(4, 1, sharex=True, figsize=(8, 6))
rcm8cube.sections['demo'].show('depth', data='spacetime',
                               ax=ax[0], label='spacetime')
rcm8cube.sections['demo'].show('depth', data='preserved',
                               ax=ax[1], label='preserved')
rcm8cube.sections['demo'].show('depth', data='stratigraphy',
                               ax=ax[2], label='quick stratigraphy')
rcm8cube.sections['demo'].show('depth', style='lines', data='stratigraphy',
                               ax=ax[3], label='quick stratigraphy')
