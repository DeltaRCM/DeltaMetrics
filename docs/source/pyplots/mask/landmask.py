"""Visual for LandMask."""
import deltametrics as dm
from deltametrics.mask import LandMask

rcm8cube = dm.sample_data.cube.rcm8()
land_mask = LandMask(rcm8cube['eta'].data[-1, :, :])
land_mask.show()
