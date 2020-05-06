import sys
import os

import matplotlib.pyplot as plt

from deltametrics.mask import ChannelMask


_arr = np.ones((50, 50))
_arr[:40, 30:36] = 0
cmsk = ChannelMask(_arr)

cmsk.show()
