from ..utils import *
from .getRfPred import *
import math
import numpy as np


def p12033(O21, O22, ms, ls, forest):
    #   This is an implemenatation of the algorithm described in 
    #	P.1203.3
    #   Inputs:
    # ====================================================
    #   I.14
    #   numStalls: number of stalling events
    #   totalBuffLen: total length of stalling events
    #   avgBuffLen: average interval between stalling events
    #
    # ====================================================
    #   O21 and O22: one score per second
    #   ms: stalling frames
    #   ls: duration of each stalling event
    #   forest: random forest as a cell array
    #   Author: Zhengfang Duanmu

    ## model parameters
    # Coefficient sets for w_buffi
    C_ref7 = 0.48412879
    C_ref8 = 10
    
    # Coefficient sets for negBias
    C1 = 1.87403625
    C2 = 7.85416481
    C23 = 0.01853820
    
    # Coefficient sets for O.34
    av1 = -0.00069084
    av2 = 0.15374283
    av3 = 0.97153861
    av4 = 0.02461776
    
    # Coefficient sets for O.35
    t1 = 0.00666620027943848
    t2 = 0.0000404018840273729
    t3 = 0.156497800436237
    t4 = 0.143179744942738
    t5 = 0.0238641564518876
    c1 = 0.67756080
    c2 = -8.05533303
    c3 = 0.17332553
    c4 = -0.01035647
    
    # Coefficient sets for O.46
    S1 = 9.35158684
    S2 = 0.91890815
    S3 = 11.0567558

    # load('randomForest.mat');

    '''total duration of the session'''
    T = len(O22)
    t = np.array(range(1, T+1))

    '''8.1.1 Parameters related to stalling'''
    numStalls = 0
    totalBuffLen = 0
    avgBuffLen = 0
    if len(ms) != 0:
        if len(ms) > 1:
            buffer_list = ms[1:] - ms[:-1]
            avgBuffLen = sum(buffer_list) / len(buffer_list)
        w_buff = C_ref7 + (1 - C_ref7) * np.exp(ms * (np.log(0.5) / (-C_ref8)))
        totalBuffLen = sum(ls * w_buff)
        numStalls = len(ms)

    # O34
    O34 = av1 + av2 * O21 + av3 * O22 + av4 * O21 * O22
    O34 = range_limit(O34, 5, 1)

    '''8.1.2 Parameters related to audiovisual quality'''
    # negativeBias
    w_diff = C1 + (1 - C1) * np.exp(-(T-t) * (np.log(0.5) / (-C2)))
    O34_diff = O34 * w_diff
    negPerc = np.percentile(O34_diff, 10, interpolation='midpoint')
    negBias = max(0, -negPerc) * C23
    # vidQualSpread
    vidQualSpread = np.max(O22) - np.min(O22)
    # vidQualChangeRate
    vidQualChangeList = abs(O22[1:] - O22[:-1])
    changeList = vidQualChangeList[vidQualChangeList > 0.2]
    vidQualChangeRate = sum(changeList) / T
    # qDirChangesTot
    maFilter = (1./5) * np.ones(5)
    O22pad = np.pad(O22, (2, 2), 'edge')
    O22MA = np.convolve(O22pad, maFilter, 'valid')
    QC_list = []
    for ppp in range(0, len(O22MA)-3, 3):
        qqq = ppp + 3
        diffMA = O22MA[ppp] - O22MA[qqq]
        if diffMA > 0.2:
            QC_list.append(1)
        elif -0.2 < diffMA < 0.2:
            QC_list.append(0)
        else:
            QC_list.append(-1)
    QC = np.array(QC_list)
    QCnoZero = QC[QC != 0]
    qDirChangesTot = 0
    if len(QCnoZero):
        qDirChangesTot = qDirChangesTot + 1
    for ppp in range(1, len(QCnoZero)):
        if QCnoZero[ppp] != QCnoZero[ppp-1]:
            qDirChangesTot = qDirChangesTot + 1
    # qDirChangesLongest
    qc_len = np.zeros((0, 2))
    distances = []
    for index in range(0, len(QC)):
        if QC[index] != 0:
            if len(qc_len):
                if qc_len[-1, 1] != QC[index]:
                    qc_len = np.append(qc_len, [[index, QC[index]]], axis=0)
            else:
                qc_len = np.array([[index, QC[index]]])
    if len(qc_len):
        qc_len = np.append([[1, 0]], qc_len, axis=0)
        qc_len = np.append(qc_len, [[len(QC), 0]], axis=0)

        for ppp in range(1, len(QC)):
            distances.append(QC[ppp] - QC[ppp-1])

        qDirChangesLongest = max(distances) * 3
    else:
        qDirChangesLongest = T

    ## 8.1.3 Parameters related to machine learning module
    mediaLength = min(len(O21), len(O22))
    reBuffCount = 0
    initBuffDur = 0
    stallDurWoIB = 0
    if len(ms):
        msWoIB = ms[ms != 0]
        reBuffCount = len(msWoIB)
        stallDurWoIB = sum(ls)

        if ms[0] == 0:
            initBuffDur = ls[0]
            stallDurWoIB = stallDurWoIB - initBuffDur

    stallDur = 1/3 * initBuffDur + stallDurWoIB
    reBuffFreq = reBuffCount / mediaLength
    stallRatio = stallDur / mediaLength
    if len(ms):
        timeLastRebuffToEnd = mediaLength - ms[-1]
    else:
        timeLastRebuffToEnd = 0

    O22_1 = O22[: int(len(O22)/3+0.5)-1]
    O22_2 = O22[int(len(O22)/3+0.5): int(len(O22)*2/3+0.5)-1]
    O22_3 = O22[int(len(O22)*2/3+0.5):]
    averagePvScoreOne = sum(O22_1) / len(O22_1)
    averagePvScoreTwo = sum(O22_2) / len(O22_2)
    averagePvScoreThree = sum(O22_3) / len(O22_3)
    onePercentilePvScore = np.percentile(O22, 1, interpolation='midpoint')
    fivePercentilePvScore = np.percentile(O22, 5, interpolation='midpoint')
    tenPercentilePvScore = np.percentile(O22, 10, interpolation='midpoint')

    O21_1 = O21[: int(len(O21)/2+0.5)-1]
    O21_2 = O21[int(len(O21)/2 + 0.5):]
    averagePaScoreOne = sum(O21_1)/len(O21_1)
    averagePaScoreTwo = sum(O21_2)/len(O21_2)

    ''' O35 and its dependencies 8-2 -- 8-11 '''
    # O35baseline
    w1 = t1 + t2 * np.exp(t / (T / t3))
    w2 = t4 - t5 * O34
    O35baseline = sum(w1 * w2 * O34) / sum(w1 * w2)
    # negBias has been computed

    # oscComp
    qDiff = max(0.1 + np.log10(vidQualSpread + 0.01), 0)
    oscTest = (qDirChangesTot / T < 0.25) & (qDirChangesLongest < 30)
    if oscTest:
        # should we remove min in Eq. 8-8? cauz it should be a penalty term
        oscComp = qDiff * np.exp(min(c1 * qDirChangesTot + c2, 1.5))
    else:
        oscComp = 0

    # adaptComp
    adaptTest = (qDirChangesTot / T < 0.25)
    if adaptTest:
        # should we remove min in Eq. 8-9? cauz it should be a penalty term
        adaptComp = c3 * vidQualSpread * vidQualChangeRate + c4
    else:
        adaptComp = 0

    # O35
    O35 = O35baseline - negBias - oscComp - adaptComp
    # Eq. 8-13
    SI = np.exp(-numStalls / S1) * np.exp(-(totalBuffLen / T) / S2) * np.exp(-(avgBuffLen / T) / S3)

    input_feature = [reBuffCount, stallDur, reBuffFreq, stallRatio,
        timeLastRebuffToEnd, averagePvScoreOne, averagePvScoreTwo,
        averagePvScoreThree, onePercentilePvScore, fivePercentilePvScore,
        tenPercentilePvScore, averagePaScoreOne, averagePaScoreTwo, mediaLength]
    RF_prediction = getRfPred(input_feature, forest)
    # Eq. 8-12
    O46 = 0.75 * (1 + (O35 - 1) * SI) + 0.25 * RF_prediction
    # Eq. 8-14
    Q = 0.02833052 + 0.98117059 * O46

    return Q, reBuffCount, stallDur, reBuffFreq, stallRatio









