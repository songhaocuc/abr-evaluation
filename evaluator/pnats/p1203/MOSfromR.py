import numpy as np


def MOSfromR(Q):
    MOS_MAX = 4.9
    MOS_MIN = 1.05
    MOS = np.zeros(len(Q))
    for i in range(len(Q)):
        if Q[i] > 0 and Q[i] < 100:
            MOS[i] = (MOS_MIN+(MOS_MAX-MOS_MIN)/100*Q[i]+Q[i]*(Q[i]-60)*(100-Q[i])*7.0e-6)
        elif Q[i] >= 100:
            MOS[i] = MOS_MAX
        else:
            MOS[i] = MOS_MIN
    return MOS
