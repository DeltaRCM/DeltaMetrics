"""Visual for WetMask."""
import deltametrics as dm
from deltametrics.mask import WetMask

rcm8cube = dm.sample_data.rcm8()
wet_mask = WetMask(rcm8cube['eta'].data[-1, :, :])
wet_mask.show()
