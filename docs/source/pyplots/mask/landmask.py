"""Visual for LandMask."""
import deltametrics as dm
from deltametrics.mask import LandMask

golfcube = dm.sample_data.golf()
land_mask = LandMask(golfcube['eta'].data[-1, :, :])
land_mask.show()
