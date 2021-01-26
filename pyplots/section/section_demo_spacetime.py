import matplotlib.pyplot as plt

import deltametrics as dm

rcm8cube = dm.sample_data.rcm8()


fig, ax = plt.subplots(figsize=(8, 2))
rcm8cube.register_section('demo', dm.section.StrikeSection(y=5))
rcm8cube.sections['demo'].show('velocity')
