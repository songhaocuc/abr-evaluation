import numpy as np
import math


def mode0(br, codRes, fps):
    # model parameters
    a1 = 11.99835
    a2 = -2.99992
    a3 = 41.24751
    a4 = 0.13183

    # core model
    bpp = br / (codRes * fps)
    quant = a1 + a2 * np.log(a3 + np.log(br) + np.log(br * bpp + a4))
    return quant