import os
import time
import random as pif

import numpy as np
from skimage.io import imread
from skimage import color
from skimage.feature import local_binary_pattern, SIFT
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from numpy import hstack, double, unique, zeros, load, asarray

from TP1.descriptors import color_histogram
from TP2.vlad import vlad


# ----- Fonction qui calcul TOUT les descripteurs pour un dataset -----------------------------------------------------

# Renvoie un dataset composer des tuples (points_clé, descriptor) et des labels
def calc_dts_sift(dataset, pb=True):
    size_dataset = len(dataset[0])
    if pb:
        return [[calc_desc_sift_pb(dataset[0], k, size_dataset) for k in range(size_dataset)], dataset[1]]
    else:
        return [[calc_desc_sift(i) for i in dataset[0]], dataset[1]]


# Renvoie un dataset composer du descripteur 'sac de mot visuel' et des labels
def calc_dts_sacmotsvisuels(dataset, path_voc, pb=True):
    vocab = load(path_voc)
    size_dataset = len(dataset[0])
    if pb:
        return [[calc_desc_sacmotsvisuels_pb(dataset[0], vocab, k, size_dataset) for k in range(size_dataset)],
                dataset[1]]
    else:
        return [[calc_desc_sacmotsvisuels(img, vocab) for img in dataset[0]], dataset[1]]


# Renvoie un dataset composer du descripteur vlad après pca et des labels
def calc_dts_vladpca(dataset, path_voc, pb=True):
    vocab = load(path_voc)
    size_dataset = len(dataset[0])
    if pb:
        x = [calc_desc_vlad_pb(dataset[0], vocab, k, size_dataset) for k in range(size_dataset)]
    else:
        x = [calc_desc_vlad(img, vocab) for img in dataset[0]]
    pca = PCA(n_components=100)
    pca.fit(x)
    return list((pca.transform(x), dataset[1]))


# Renvoie un dataset composer des descripteurs vlad concaténer d'une image diviser en région
def calc_dts_vladgeo(dataset, n_split, path_voc, ):
    vocab = load(path_voc)
    size_dataset = len(dataset[0])
    return [np.asarray(calc_desc_geocutingvlad_pb(dataset[0], n_split, vocab, k, size_dataset)).flatten()
            for k in range(size_dataset)], dataset[1]


# ----------------------------------------------------------------------------------------------------------------------


# ----- Fonction qui renvoie UN descripteur pour UNE image -------------------------------------------------------------


# Renvoie le descripteur correspondant à l'histogramme de l'image
def calc_desc_histogramme(image):
    return color_histogram(image)


# Renvoie le descripteur correspondant à l'histogramme LBP de l'image
def calc_desc_histolbp(image, nb_points, radius):
    if len(image.shape) != 3:
        img_nb = image
    else:
        img_nb = color.rgb2gray(image)
    lbp = local_binary_pattern(img_nb, nb_points, radius)
    return calc_desc_histogramme(lbp)


# Renvoie le descripteur correspondant à la concatenation de l'histogramme de l'image et de l'histogramme lbp
def calc_desc_histo_histolbp(image, nb_points, radius):
    return hstack((calc_desc_histogramme(image),
                   calc_desc_histolbp(image, nb_points, radius)
                   ))


# Renvoie le descripteur correspondant a la concatenation des histogrammes de chaque partie de l'image
def calc_desc_geocuting(image, nb_split):
    ps_img = get_parts_of_image(image, nb_split)
    res = []
    for p in ps_img:
        res = hstack((res, calc_desc_histogramme(p)))
    return res


# Renvoie le descripteur correspondant à la concatenation des vlad de chaque partie de l'image
def calc_desc_geocutingvlad_pb(dts, nb_split, vocab, k, nbi):
    ps_img = get_parts_of_image(dts[k], nb_split)
    progressbar(k / (nbi - 1))
    return [calc_desc_vlad(p, vocab) for p in ps_img]


# Renvoie le descripteur correspondant à l'utilisation de la methode SIFT
def calc_desc_sift(imgnb):
    sift = SIFT()
    try:
        sift.detect_and_extract(imgnb)
        keypoints = sift.keypoints
        desccriptor = sift.descriptors
        return keypoints, desccriptor
    except RuntimeError:
        return [], np.zeros((1, 128))


# Renvoie le descripteur correspondant à l'utilisation de la methode SIFT (avec progress bar)
def calc_desc_sift_pb(dts, k, nbi):
    progressbar(k / (nbi - 1))
    return calc_desc_sift(dts[k])


# Renvoie le descripteur correspondant à l'histogramme des mots visuels de l'image
def calc_desc_sacmotsvisuels(img, vocab):
    key_points, descripteur = calc_desc_sift(img)
    kmeans = KMeans(n_clusters=len(vocab), random_state=42)
    kmeans.fit(descripteur)
    kmeans.cluster_centers_ = vocab.astype(double)
    pred = kmeans.predict(descripteur)
    count_n = unique(pred, return_counts=True)
    histo = [0 if idx not in count_n[0] else count_n[1][list(count_n[0]).index(idx)] for idx in range(len(vocab))]
    return histo


# Renvoie le descripteur correspondant à l'utilisation des VLAD
def calc_desc_vlad(img, vocab):
    _, desc = calc_desc_sift(img)
    return vlad(desc, vocab.astype(double))


# Renvoie le descripteur correspondant à l'histogramme des mots visuels de l'image (avec progress bar)
def calc_desc_sacmotsvisuels_pb(dts, v, k, nbi):
    progressbar(k / (nbi - 1))
    return calc_desc_sacmotsvisuels(dts[k], v)


# Renvoie le descripteur correspondant à l'utilisation des VLAD (avec progress bar)
def calc_desc_vlad_pb(dts, v, k, nbi):
    progressbar(k / (nbi - 1))
    return calc_desc_vlad(dts[k], v)


# ----------------------------------------------------------------------------------------------------------------------


# ----- Fonction d'affichage & de visualisation ------------------------------------------------------------------------

# Permet d'afficher la matrice de confusion dans une figure et de la sauvegarder
def afficher_confmat(cm, classes, save_name):
    dsp_mat = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    dsp_mat.plot()
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(save_name)
    plt.show()


# Créé 'n' figures composer chacune d'une image parmi 'imgs' et de ces points clés correspondant dans 'sift_res'
def afficher_n_images_keypoints(n, imgs, sifts_res):
    rand_imgs = list(zip(imgs, sifts_res))
    pif.Random(42).shuffle(rand_imgs)
    if n > len(rand_imgs): return
    for k in range(n):
        afficher_keypoints(rand_imgs[k][0], rand_imgs[k][1])


# Créé une figure avec l'image et les point clés correspondantes superposées à celle-ci
def afficher_keypoints(img, sift_res):
    plt.figure(time.time_ns())
    plt.imshow(img, cmap='gray', vmin=0, vmax=1)
    plt.scatter(asarray(sift_res[0])[:, 1], asarray(sift_res[0])[:, 0], s=2)


# Affiche une barre de progression remplie à 'percent' pour cent
def progressbar(percent, size=50):
    nb_complet = int(percent * size)
    barc = ["█" for _ in range(nb_complet)]
    barv = ["." for _ in range(size - nb_complet)]
    bar = ""
    print("\r\r", end='')
    print(f"{bar.join(barc) + bar.join(barv)} : {(percent * 100):.2f}%", end='')


# ----------------------------------------------------------------------------------------------------------------------


# ----- Fonction generale ou intermediaire -----------------------------------------------------------------------------

# Fonction generale qui entraine et évalue un modèle, renvoie la precision et la matrice de confusion
def training_process_and_eval(model, data, nb_classes=None, ratio=0.8, rs=42):
    x_train, x_test, y_train, y_test = train_test_split(data[0], data[1], test_size=1 - ratio, random_state=rs)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    acc = accuracy_score(y_test, y_pred)
    mat_conf = confusion_matrix(y_test, y_pred, labels=range(nb_classes))
    return acc, mat_conf


# Fonction générale qui renvoi un dataset d'image en noir et blanc avec leur label
def path_file_to_greyset(path_to_data):
    colorset, n_classes, files_names = path_file_to_colorset(path_to_data)
    auto_rgb = lambda img: img if len(img.shape) != 3 else color.rgb2gray(img)
    Xg = [auto_rgb(img) for img in colorset[0]]
    greyset = [Xg, colorset[1]]
    return greyset, n_classes, files_names


# Fonction générale qui renvoi un dataset d'image en couleur avec leur label
def path_file_to_colorset(path_to_data):
    rand_seed = 42
    listed_files = os.listdir(path_to_data)
    pif.Random(rand_seed).shuffle(listed_files)
    X = [imread(f"./{path_to_data}/{file}") for file in listed_files]
    y_name = [fname.split("_")[0] for fname in listed_files]
    n_classes = list(set(y_name))
    n_classes.sort()
    y = [n_classes.index(n) for n in y_name]
    data_from_images = [X, y]
    return data_from_images, n_classes, listed_files


# Fonction intermediaire qui permet de recuperé un sous tableau de taille size qui commence en x, y
def get_sub_array(arr, x, y, size):
    sa = arr[x:x + size[0], y:y + size[1]]
    if len(arr.shape) > 2:
        res = zeros((size[0], size[1], arr.shape[2]))
    else:
        res = zeros((size[0], size[1]))

    for l in range(sa.shape[0]):
        for w in range(sa.shape[1]):
            res[l, w] = sa[l, w]
    return res.astype('int')


# Fonction intermediaire qui permet de découper une image en nb_split ligne et colonne
def get_parts_of_image(image, nb_split):
    nb_w = int(image.shape[0] / nb_split) + 1
    nb_h = int(image.shape[1] / nb_split) + 1
    parts = []
    for lig in range(nb_split):
        for col in range(nb_split):
            p = get_sub_array(image, lig * nb_w, col * nb_h, (nb_w, nb_h))
            parts.append(p)
    return parts
