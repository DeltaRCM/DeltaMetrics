"""Visual for ShoreMask."""
import deltametrics as dm
from deltametrics.mask import ShorelineMask

rcm8cube = dm.sample_data.rcm8()
shore_mask = ShorelineMask(rcm8cube['eta'].data[-1, :, :])
shore_mask.show()
