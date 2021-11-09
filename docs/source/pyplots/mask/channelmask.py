"""Visual for ChannelMask."""
import deltametrics as dm
from deltametrics.mask import ChannelMask

golfcube = dm.sample_data.golf()
channel_mask = ChannelMask(rcm8cube['velocity'].data[-1, :, :],
                           rcm8cube['eta'].data[-1, :, :])
channel_mask.show()
