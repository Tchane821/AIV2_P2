import math
import os

import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from skvideo.io import vread
from skimage.registration import optical_flow_tvl1
from skimage.color import rgb2gray, hsv2rgb

from Tools.mytoolsbox import calc_desc_histogramme, progressbar
from Tools.setup import DATA_VIDEO_1_PATH, RAND_SEED, EXPORT_DATA_FLUXOPT, DATA_VIDEO_2_VOC_PATH, DATA_VIDEO_2_KEY_PATH
from numpy import zeros, empty, arctan2, pi, asarray, save, linalg, load, unique, double
from math import sqrt, pow
import random as pif

from stip import read_stip_file


# TP4 PARTIE 1 HISTOGRAMME AVG -----------------------------------------------------------------------------------------

# Renvoie le dataset, les N classes et la liste des fichiers avec en desc les histogrammes moyen
def path_file_to_video_datasets_histiavg(paths_videos):
    listed_videos = os.listdir(paths_videos)
    pif.Random(RAND_SEED).shuffle(listed_videos)
    X = calc_dtsv_histoavg(listed_videos)
    y_name = [fname.split("_")[0] for fname in listed_videos]
    n_classes = list(set(y_name))
    n_classes.sort()
    y = [n_classes.index(n) for n in y_name]
    return (X, y), n_classes, listed_videos


# Calcule et renvoie pour chaque video son histogramme moyen
def calc_dtsv_histoavg(paths_video):
    def tpb(data, prog):
        progressbar(prog)
        return calc_desc_avg_histogrames(data)

    return [tpb(paths_video[k], k / (len(paths_video) - 1)) for k in range(len(paths_video))]


# Calcule et renvoie pour une video l'histogramme moyen
def calc_desc_avg_histogrames(path_video) -> []:
    hs = calc_frames_histogrammes(path_video)
    hlen = len(hs[1])
    hsum = zeros(hlen)
    for h in hs:
        hsum = [hsum[k] + h[k] for k in range(hlen)]
    map(lambda x: x / hlen, hsum)
    return hsum


# Calcule pour une video tout les histogrammes de couleur par image
def calc_frames_histogrammes(path_video) -> []:
    video = vread(f"{DATA_VIDEO_1_PATH}/{path_video}")
    res = [calc_desc_histogramme(f) for f in video]
    return res


# TP4 PARTIE 2 VISUALISATION FLUX OPTIQUES -----------------------------------------------------------------------------

# Affiche l'image k d'une video
def afficher_image_k(fo_rgb, k):
    img1 = fo_rgb[k] * 255
    img1 = img1.astype(int)
    plt.figure()
    plt.imshow(img1)


# Sauvegarde un numpy array correspondant a une video
def save_video_numpyarray(arr, name):
    save(f"{EXPORT_DATA_FLUXOPT}/{name}.npy", arr)


# Convertie une suite d'image HSV en une suite d'image RGB
def conv_fo_hsv_2_fo_rgb(fo_hsv):
    return asarray([hsv2rgb(i) for i in fo_hsv])


# Converti les flux optiques en images au format hsv
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


# Calcule les flux optique d'une video
def calc_opticalflux(path_video) -> []:
    video_color = vread(f"{DATA_VIDEO_1_PATH}/{path_video}")
    video_wb = [rgb2gray(i) for i in video_color]

    def opf_pb(k):
        progressbar(k / (len(video_wb) - 2))
        return optical_flow_tvl1(video_wb[k], video_wb[k + 1])

    return my_reshape(asarray([opf_pb(k) for k in range(len(video_wb) - 1)]))


# Un reshape qui me permet de passez de
# idx_image, idx_tuple, idx_ligne, idx_colo) en (idx_image, idx_ligne, idx_colo, idx_tuple)
# Avec idx_tuple = les infos du flux optique sur x et y
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


# TP4 PARTIE 3 FO ORIENTATIONS ET MAGNITUDE ----------------------------------------------------------------------------

# Transforme les flux optiques en flux optique normaliser de 0 à 1
def calc_fo_2_tuple(optflx):
    flo_om = zeros((optflx.shape[0], optflx.shape[1], optflx.shape[2], 3), 'int8')
    flo_om[..., 0] = (arctan2(optflx[..., 1], optflx[..., 0]) / pi * 180. + 180.).astype(int)
    flo_om[..., 1] = linalg.norm(optflx, axis=3, ord=2).astype(int)
    return flo_om


def extract_firstmagn_perclass():
    pass


# TP5 PARTIE 2 CLASSIFICATION ------------------------------------------------------------------------------------------

# Créé le dataset tout bo tout propre pour l'entrainement avec les descripteur HOG, HOF ou HOGHOF
def path_file_to_video_datasets_hog_hof(paths_videos, ho="HOGHOF"):
    listed_videos = os.listdir(paths_videos)
    pif.Random(RAND_SEED).shuffle(listed_videos)
    if ho == "HOG":
        print("\tLog: Descripteurs choisie : HOG")
        X = calc_dts_vecteur_freq(listed_videos, calc_desc_vecteur_freq_hog)
    elif ho == "HOF":
        print("\tLog: Descripteurs choisie : HOF")
        X = calc_dts_vecteur_freq(listed_videos, calc_desc_vecteur_freq_hof)
    else:
        print("\tLog: Descripteurs choisie : HOGHOF")
        X = calc_dts_vecteur_freq(listed_videos, calc_desc_vecteur_freq_hof)
    print()
    y_name = [fname.split("_")[0] for fname in listed_videos]
    n_classes = list(set(y_name))
    n_classes.sort()
    y = [n_classes.index(n) for n in y_name]
    return (X, y), n_classes, listed_videos


# Calcule pour chaque fichier le descripteur correspondant avec desc = func(fichier[k]) avec k => [0,nb_fichier-1]
def calc_dts_vecteur_freq(files_names, func):
    descs = []
    nb_file = len(files_names)
    for k in range(nb_file):
        descs.append(func(files_names[k]))
        progressbar(k / (nb_file - 1))
    return descs


# Fonction qui calcule l'histogramme des vecteurs de fréquences pour HOGHOF
def calc_desc_vecteur_freq_hoghof(file_name):
    kp, desc = read_stip_file(f"{DATA_VIDEO_2_KEY_PATH}/{file_name.split('.')[0]}.key")
    vocab = load(f"{DATA_VIDEO_2_VOC_PATH}/voc_hoghof_500.npy")
    return calc_kmeans_histo(desc, vocab)


# Fonction qui calcule l'histogramme des vecteurs de fréquences pour HOG
def calc_desc_vecteur_freq_hog(file_name):
    kp, desc = read_stip_file(f"{DATA_VIDEO_2_KEY_PATH}/{file_name.split('.')[0]}.key")
    desc_hog = desc[:, :72]
    vocab = load(f"{DATA_VIDEO_2_VOC_PATH}/voc_hog_500.npy")
    return calc_kmeans_histo(desc_hog, vocab)


# Fonction qui calcule l'histogramme des vecteurs de fréquences pour HOF
def calc_desc_vecteur_freq_hof(file_name: str):
    kp, desc = read_stip_file(f"{DATA_VIDEO_2_KEY_PATH}/{file_name.split('.')[0]}.key")
    desc_hof = desc[:, 72:]
    vocab = load(f"{DATA_VIDEO_2_VOC_PATH}/voc_hof_500.npy")
    return calc_kmeans_histo(desc_hof, vocab)


# Calcule histogramme correspondant au descripteur et au vocabulaire fournit
def calc_kmeans_histo(desc, vocab):
    kmeans = KMeans(n_clusters=5, random_state=42)
    kmeans.fit(desc.astype(float))
    kmeans.cluster_centers_ = vocab.astype(float)
    pred = kmeans.predict(desc.astype(float))
    count_n = unique(pred, return_counts=True)
    histo = [0 if idx not in count_n[0] else count_n[1][list(count_n[0]).index(idx)] for idx in range(len(vocab))]
    return histo


def afficher_video_keypoint(path_file):
    pass
