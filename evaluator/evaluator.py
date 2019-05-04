from .pnats.P1203 import *
import numpy as np
from scipy.io import loadmat
import os
import json

def preprocess(args):
    '''处理传入参数'''
    # audio_bitrate = np.ones(20) * 40
    # video_bitrate = np.ones(20) * 1200
    # disRes = np.ones(20) * 1920 * 1080
    # codRes = np.ones(20) * 1920 * 1080
    # fps = np.ones(20) * 25
    # handheal = False
    # ms = []
    # ls = []
    '''解析'''

    # arguments = json.loads(args)
    bitrate_list = args['play_config']['bitrate_list']

    play_trace = args['play_trace']
    video_bitrate = np.array(play_trace['video_bitrate'])
    audio_bitrate = np.array(play_trace['audio_bitrate'])

    index_list = play_trace['index']
    disRes = []
    codRes = []
    fps = []
    for i in range(len(index_list)):
        rep_index = index_list[i]
        rep_info = bitrate_list[rep_index]
        disRes.append(1920 * 1080)
        codRes.append(rep_info['width'] * rep_info['height'])
        fps.append(25)
    disRes = np.array(disRes)
    codRes = np.array(codRes)
    fps = np.array(fps)


    handheld = False
    '''测试数据'''
    '''
    audio_bitrate = np.ones(20) * 40
    vl = [1,1,1,1,1,9,9,9,9,9,1,1,1,1,1,6,6,6,6,6]
    video_bitrate = np.array(vl) * 500
    disRes = np.ones(20) * 1920 * 1080
    codRes = np.ones(20) * 1920 * 1080
    fps = np.ones(20) * 25
    handheld = False
    ms = []
    ls = []
    '''
    '''结束'''

    stall_record = args['stall_record']
    ms = stall_record['position']
    ls = stall_record['duration']

    ms = np.array(ms)
    ls = np.array(ls)


    return audio_bitrate, video_bitrate, disRes, codRes, fps, handheld, ms, ls


def evaluate(args):
    audio_bitrate, video_bitrate, disRes, codRes, fps, handheld, ms, ls = preprocess(args)

    filepath = os.path.dirname(__file__) + '/randomForest.mat'
    m = loadmat(filepath)
    forest = m['forest']

    Q, rebuff_count, stall_dur, rebuff_freq, stall_ratio = p1203(audio_bitrate, video_bitrate, disRes, codRes, fps, handheld, ms, ls, forest)

    # try:
    #     Q = p1203(audio_bitrate, video_bitrate, disRes, codRes, fps, handheld, ms, ls, forest)
    # except BaseException:
    #     print("[EXCEPTION]")
    #     Q = 0
    # else:
    #     pass

    aver_vbr = sum(video_bitrate)/ len(video_bitrate)
    aver_abr = sum(audio_bitrate)/ len(audio_bitrate)
    response = dict()
    response["QoE"] = Q
    response["rebuff_count"] = rebuff_count
    response["stall_dur"] = stall_dur
    response["rebuff_freq"] = rebuff_freq
    response["stall_ratio"] = stall_ratio
    response["aver_vbr"] = aver_vbr
    response["aver_abr"] = aver_abr

    return response
