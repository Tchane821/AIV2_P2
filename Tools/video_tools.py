import os

import numpy as np
from skvideo.io import vread
from skimage.registration import optical_flow_tvl1
from skimage.color import rgb2gray

from Tools.mytoolsbox import calc_desc_histogramme, progressbar
from Tools.setup_env import DATA_VIDEO_PATH, RAND_SEED
from numpy import zeros
import random as pif


# TP4 PARTIE 1 HISTOGRAMME AVG -----------------------------------------------------------------------------------------

def path_file_to_video_datasets_histiavg(paths_videos):
    listed_videos = os.listdir(paths_videos)
    pif.Random(RAND_SEED).shuffle(listed_videos)
    X = calc_dtsv_histoavg(listed_videos)
    y_name = [fname.split("_")[0] for fname in listed_videos]
    n_classes = list(set(y_name))
    n_classes.sort()
    y = [n_classes.index(n) for n in y_name]
    return (X, y), n_classes, listed_videos


def calc_dtsv_histoavg(paths_video):
    def tpb(data, prog):
        progressbar(prog)
        return calc_desc_avg_histogrames(data)

    return [tpb(paths_video[k], k / (len(paths_video) - 1)) for k in range(len(paths_video))]


def calc_desc_avg_histogrames(path_video) -> []:
    hs = calc_frames_histogramlmmes(path_video)
    hlen = len(hs[1])
    hsum = zeros(hlen)
    for h in hs:
        hsum = [hsum[k] + h[k] for k in range(hlen)]
    map(lambda x: x / hlen, hsum)
    return hsum


def calc_frames_histogramlmmes(path_video) -> []:
    video = vread(f"{DATA_VIDEO_PATH}/{path_video}")
    res = [calc_desc_histogramme(f) for f in video]
    return res


# TP4 PARTIE 2 VISUALISATION FLUX OPTIQUES

def calc_opticalflux(path_video) -> []:
    video_color = vread(f"{DATA_VIDEO_PATH}/{path_video}")
    video_wb = [rgb2gray(i) for i in video_color]

    def opf_pb(k):
        progressbar(k / (len(video_wb) - 2))
        return optical_flow_tvl1(video_wb[k], video_wb[k + 1])

    return np.asarray([opf_pb(k) for k in range(len(video_wb) - 1)])
