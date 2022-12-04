import math
import os

import matplotlib.pyplot as plt
from skvideo.io import vread
from skimage.registration import optical_flow_tvl1
from skimage.color import rgb2gray, hsv2rgb

from Tools.mytoolsbox import calc_desc_histogramme, progressbar
from Tools.setup import DATA_VIDEO_PATH, RAND_SEED, EXPORT_DATA_FLUXOPT
from numpy import zeros, empty, arctan2, pi, asarray, save
from math import sqrt, pow
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


# TP4 PARTIE 2 VISUALISATION FLUX OPTIQUES -----------------------------------------------------------------------------

def afficher_image_k(fo_rgb, k):
    img1 = fo_rgb[k] * 255
    img1 = img1.astype(int)
    plt.figure()
    plt.imshow(img1)


def save_video_numpyarray(arr, name):
    save(f"{EXPORT_DATA_FLUXOPT}/{name}.npy", arr)


def conv_fo_hsv_2_fo_rgb(fo_hsv):
    return asarray([hsv2rgb(i) for i in fo_hsv])


def conv_optical_flux_2_hsv(optflx):
    flo_hsv = empty((optflx.shape[0], optflx.shape[1], optflx.shape[2], 3), 'float32')
    max_m = 0
    min_m = math.inf
    max_d = 0
    min_d = math.inf
    # Pour chaque image, pour chaque ligne, pour chaque colone
    for i in range(optflx.shape[0]):
        for l in range(optflx.shape[1]):
            for c in range(optflx.shape[2]):
                # On calcule la direction du vecteur dx, dy
                direction = arctan2(optflx[i, l, c, 1], optflx[i, l, c, 0])
                max_d = max(max_d, direction)  # On stock max et min
                min_d = min(min_d, direction)
                flo_hsv[i, l, c, 0] = direction
                flo_hsv[i, l, c, 1] = 1  # le chan V = 1
                # On calcule la norme de ce vecteur
                magnitude = sqrt(pow(optflx[i, l, c, 0], 2) + pow(optflx[i, l, c, 1], 2))
                max_m = max(max_m, magnitude)  # On stock le min et max
                min_m = min(min_m, magnitude)
                flo_hsv[i, l, c, 2] = magnitude
        progressbar(i / (optflx.shape[0] - 1))  # Ne fait que de l'affichage
    # Normalisation de "axe 3"
    # On normalise la norme et la direction
    print("\n\tLog: Normalisation: in progress...")
    for i in range(optflx.shape[0]):
        for l in range(optflx.shape[1]):
            for c in range(optflx.shape[2]):
                flo_hsv[i, l, c, 0] = (flo_hsv[i, l, c, 0] - min_d) / (max_d - min_d)
                flo_hsv[i, l, c, 2] = (flo_hsv[i, l, c, 2] - min_m) / (max_m - min_m)
    print("\tLog: Normalisation: done!")
    return flo_hsv


def calc_opticalflux(path_video) -> []:
    video_color = vread(f"{DATA_VIDEO_PATH}/{path_video}")
    video_wb = [rgb2gray(i) for i in video_color]

    def opf_pb(k):
        progressbar(k / (len(video_wb) - 2))
        return optical_flow_tvl1(video_wb[k], video_wb[k + 1])

    return my_reshape(asarray([opf_pb(k) for k in range(len(video_wb) - 1)]))


def my_reshape(raw_optical_flux):
    print("\n\tLog: Reshape: in progress...")
    rof_shape = raw_optical_flux.shape
    optf = empty((rof_shape[0], rof_shape[2], rof_shape[3], rof_shape[1]), 'float32')
    for idx_image in range(rof_shape[0]):
        for idx_tuple in range(rof_shape[1]):
            for idx_ligne in range(rof_shape[2]):
                for idx_colo in range(rof_shape[3]):
                    optf[idx_image, idx_ligne, idx_colo, idx_tuple] = raw_optical_flux[
                        idx_image, idx_tuple, idx_ligne, idx_colo]
    print("\tLog: Reshape: done!")
    return optf
