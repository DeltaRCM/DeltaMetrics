"""Visual for WetMask."""
import deltametrics as dm
from deltametrics.mask import WetMask

golfcube = dm.sample_data.golf()
wet_mask = WetMask(golfcube['eta'].data[-1, :, :])
wet_mask.show()
