import math
from .MOSfromR import *
import numpy as np


def p12032(audio_bitrate, audio_codec_idx):
    # audio_codec_idx table:
    # 1: MPEG1 L2
    # 2: AC3
    # 3: ACC-LC
    # 4: HE-AAC v2
    ############################################
    audio_bitrate = np.array(audio_bitrate)

    # model parameters
    a1A = 100
    if audio_codec_idx == 1:
        a2A = -0.02
        a3A = 15.48
    elif audio_codec_idx == 2:
        a2A = -0.03
        a3A = 15.70
    elif audio_codec_idx == 3:
        a2A = -0.05
        a3A = 14.60
    else:
        a2A = -0.11
        a3A = 20.06

    QcodA = a1A * np.exp(a2A * audio_bitrate) + a3A
    QA = 100 - QcodA
    O21 = MOSfromR(QA)
    return O21
