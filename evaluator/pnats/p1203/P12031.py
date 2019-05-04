from .mode0 import mode0
from ..utils import *
from .MOSfromR import *
from .RfromMOS import *
import numpy as np
import math
import sys


def p12031(bitrate, disRes, codRes, fps, handheld):
    # This is an implementation of ITU P.1203.1
    ###############################################

    # model parameters
    # 8.1.1 parameters
    q1 = 4.66
    q2 = -0.07
    q3 = 4.06

    # 8.1.2 parameters
    u1 = 72.61
    u2 = 0.32

    # 8.1.3 parameters
    t1 = 30.98
    t2 = 1.29
    t3 = 64.65

    # device parameters
    htv1 = -0.60293
    htv2 = 2.12382
    htv3 = -0.36936
    htv4 = 0.03409

    ## model implementation
    O22 = np.zeros(len(bitrate))
    # 8.1.1 Quantization degradation
    quant = mode0(bitrate, codRes, fps)
    MOSq = q1 + q2 * np.exp(q3 * quant)
    # MOSq = max(min(MOSq, 5), 1)
    range_limit(MOSq, 5, 1)  # _new_

    Dq = 100 - RfromMOS(MOSq)
    # Dq = max(min(Dq, 100), 0)
    range_limit(Dq, 100, 0)  # _new_

    # 8.1.2 Upscaling degradation
    # scaleFactor = max(disRes / codRes, 1)
    scaleFactor = range_limit(disRes / codRes, sys.maxsize, 1)
    Du = u1 * np.log10(u2 * (scaleFactor - 1) + 1)
    # Du = max(min(Du, 100), 0)
    range_limit(Du, 100, 0)  # _new_

    # 8.1.3 Temporal degradation
    Dt1 = 100 * (t1 - t2 * fps) / (t3 + fps)
    Dt2 = Dq * (t1 - t2 * fps) / (t3 + fps)
    Dt3 = Du * (t1 - t2 * fps) / (t3 + fps)

    Dt = np.zeros(len(Dq))
    index_fps24 = np.where(fps < 24)
    Dt[index_fps24] = Dt1[index_fps24] - Dt2[index_fps24] - Dt3[index_fps24]
    # Dt = max(min(Dt, 100), 0)
    range_limit(Dt, 100, 0) # _new_

    # 8.1.4 Integration
    # D = max(min(Dq + Du + Dt, 100), 0)
    D = range_limit(Dq + Du + Dt, 100, 0) # _new_

    Q = 100 - D
    mos_from_r = MOSfromR(Q)
    for i in range(len(Dq)):
        if Du[i] == 0 and Dt[i] == 0:
            O22[i] = MOSq[i]
        else:
            O22[i] = mos_from_r[i]

    if handheld:
        MOSqh = htv1 + htv2 * O22 + htv3 * O22^2 + htv4 * O22^3
        O22 = range_limit(MOSqh, 5, 1)

    return O22
