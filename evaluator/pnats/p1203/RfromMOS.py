from ..utils import *
import numpy as np
import math


def RfromMOS(MOSarg):
    MOS = np.array(MOSarg)
    range_limit(MOS, 4.5, 0)
    Q = np.zeros(len(MOS))
    h = np.zeros(len(MOS))

    for i in range(len(MOS)):
        if MOS[i] > 2.7505 :
            h[i] = (1./3) * (math.pi - math.atan(15 * math.sqrt(-903522 + 1113960 * MOS[i] - 202500 * MOS[i]**2) / (6750 * MOS[i] - 18566)))
        else:
            h[i] = (1./3) * (math.atan(15 * math.sqrt(-903522 + 1113960 * MOS[i] - 202500 * MOS[i]**2) / (6750 * MOS[i] - 18566)))

    Q = 20 * (8 - math.sqrt(226) * np.cos(h + (math.pi / 3))) / 3
    return Q
