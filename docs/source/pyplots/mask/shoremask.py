"""Visual for ShoreMask."""
import deltametrics as dm
from deltametrics.mask import ShorelineMask

rcm8cube = dm.sample_data.cube.rcm8()
shore_mask = ShorelineMask(rcm8cube['eta'][-1, :, :])
shore_mask.show()
