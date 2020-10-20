"""Visual for ChannelMask."""
import deltametrics as dm
from deltametrics.mask import ChannelMask

rcm8cube = dm.sample_data.cube.rcm8()
channel_mask = ChannelMask(rcm8cube['velocity'].data[-1, :, :],
                           rcm8cube['eta'].data[-1, :, :])
channel_mask.show()
