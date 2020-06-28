"""Visual for ShoreMask."""
import deltametrics as dm
from deltametrics.mask import ShoreMask

rcm8cube = dm.sample_data.cube.rcm8()
shore_mask = ShoreMask(rcm8cube['eta'][-1, :, :])
shore_mask.show()
