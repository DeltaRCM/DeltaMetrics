import matplotlib.pyplot as plt

import deltametrics as dm

golfcube = dm.sample_data.golf()


fig, ax = plt.subplots(figsize=(8, 2))
golfcube.register_section('demo', dm.section.StrikeSection(idx=5))
golfcube.sections['demo'].show('velocity')
