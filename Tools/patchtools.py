import math

import matplotlib.pyplot as plt
import numpy as np
from sklearn.feature_extraction.image import extract_patches_2d
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from pickle import dump, load as load_pkl
from numpy import zeros, asarray, save, load
from Tools.setup import EXPORT_DATA_PATH, EXPORT_MODELE_PATH, EXPORT_PATCHES_PATH, \
    EXPORT_SCALER_PATH, EXPORT_DATA_HARD, EXPORT_DATA_SOFT, EXPORT_DATA_CONV
from Tools.mytoolsbox import get_sub_array, progressbar


# Pour un ficher de pactches de une image créé son hard-assigment
def hard_assignment(path_to_paches, mod_nbpach, mod_nbcentro):
    patches = load(f"{EXPORT_DATA_PATH}/{path_to_paches}")
    patch_size = path_to_paches.split("_")[2].split(".")[0]
    f_name = path_to_paches.split("_")[0]
    kmod = load_pkl(open(f"{EXPORT_MODELE_PATH}/model_{patch_size}_{mod_nbpach}_{mod_nbcentro}.pkl", 'rb'))
    res = kmod.predict(patches)
    save(f"{EXPORT_DATA_HARD}/{f_name}_hard_{patch_size}.npy", asarray(res))


# Pour un ficher de pactches de une image créé son soft-assigment
def soft_assignment(path_to_paches, mod_nbpach, mod_nbcentro):
    patches = load(f"{EXPORT_DATA_PATH}/{path_to_paches}")
    patch_size = path_to_paches.split("_")[2].split(".")[0]
    f_name = path_to_paches.split("_")[0]
    kmod = load_pkl(open(f"{EXPORT_MODELE_PATH}/model_{patch_size}_{mod_nbpach}_{mod_nbcentro}.pkl", 'rb'))
    res_dp = kmod.transform(patches)
    # Distance fouareuse
    res_d = [sum([abs(x) for x in dp]) for dp in res_dp]
    res = [1 / (1 - math.exp(-d)) for d in res_d]
    save(f"{EXPORT_DATA_SOFT}/{f_name}_soft_{patch_size}.npy", asarray(res))


# Permet d'afficher les centroids d'un modèle
def afficher_patches_centroids(path_to_model):
    kmeans = load_pkl(open(f"{EXPORT_MODELE_PATH}/{path_to_model}", 'rb'))
    clusters_raw = kmeans.cluster_centers_
    size_of_cluster_patche, number_of_patch = int(path_to_model.split('_')[1]), int(path_to_model.split('_')[2])

    scaler_file_name = f"scaler_{size_of_cluster_patche}_{number_of_patch}.pkl"
    scaler = load_pkl(open(f"{EXPORT_SCALER_PATH}/{scaler_file_name}", 'rb'))

    clusters = scaler.inverse_transform(clusters_raw)

    for k in range(len(clusters)):
        plt.figure(figsize=(4, 4))
        plt.axis('off')
        img_color = np.reshape(clusters[k], (size_of_cluster_patche, size_of_cluster_patche, 3))
        img_color = img_color.astype(int)
        plt.imshow(img_color)
    plt.show()


# Permet de process un model Kmeans saur les images avec nb_center nombre de centroids
def kmeans_save_from_patches(paths_to_data: [str], nb_center: int, rs=42):
    print(f"Log: Calcule des modèles de centroids {nb_center}: in progress...")
    for pth in paths_to_data:
        data = load(f"{EXPORT_PATCHES_PATH}/{pth}")
        model = KMeans(n_clusters=nb_center, random_state=rs)
        model.fit(data)
        dump(model, open(
            f"{EXPORT_MODELE_PATH}/model_{pth.split('_')[1]}_{pth.split('_')[2].split('.')[0]}_{nb_center}.pkl",
            'wb'))
    print(f"Log: Calcule des modèles de {nb_center} centroids: done!")


# Permet de sauvegarder tout les patches normaliser et calculer pour chaque image
def compute_and_save_patches(imgs, w, nbp):
    def extpbp(dts, k):
        progressbar(k / (len(dts) - 1))
        return extract_patches(dts[k], w, nbp)

    print(f"Log: Extraction de {nbp} patchs de taille {w}: in progress...")
    r = [extpbp(imgs, k) for k in range(len(imgs))]
    print()
    rc = compact_patches(r)
    scaler = StandardScaler()
    scaler.fit(rc)
    dump(scaler, open(f"{EXPORT_SCALER_PATH}/scaler_{w}_{nbp}.pkl", "wb"))
    r_norme = scaler.transform(rc)
    save(f"{EXPORT_PATCHES_PATH}/patches_{w}_{nbp}.npy", asarray(r_norme))
    print(f"Log: Extraction de {nbp} patchs de taille {w}: done!")


# Permet de renvoyer les nbp patches de taille w de l'image img
def extract_patches(img, w: int, nbp: int, rs=42):
    if len(img.shape) < 3:
        img = imagegrey_to_3dgreyimage(img)
    patches = extract_patches_2d(img, (w, w), max_patches=nbp, random_state=rs)
    return patches


# Permet de renvoyer tout les pache de taille w/w d'une image img
def extract_patch_with_out_cover(img, w):
    if len(img.shape) < 3:
        img = imagegrey_to_3dgreyimage(img)
    r = []
    lig = int(img.shape[0] / w)
    col = int(img.shape[1] / w)
    for line in range(lig):
        for row in range(col):
            r.append(get_sub_array(img, line * w, row * w, (w, w)))
    return r


# Permet de sauvegarder tout les paches de taille w d'une images sans recouvrement pour chaque images dans imgs
def compute_and_save_allpatches_with_out_cover(dataset, w, files_names):
    dts_size = len(dataset[0])

    def extwpb(dts, k):
        progressbar(k / (dts_size - 1))
        return extract_patch_with_out_cover(dts[k], w)

    print("Log: Extraction des patches: in progress...")
    r = [extwpb(dataset[0], k) for k in range(dts_size)]
    print()
    print("Log: Extraction des patches: done!")

    print("Log: Processing normalisation des patchs d'image: in progress...")
    rc = compact_patches(r)
    scaler = StandardScaler()
    scaler.fit(rc)
    dump(scaler, open(f"{EXPORT_DATA_PATH}/scaler_forall_{w}.pkl", "wb"))
    rcp = [scaler.transform([p.flatten() for p in imgps]) for imgps in r]
    print("Log: Processing normalisation des patchs d'image: done!")

    print("Log: Sauvegarde des pathces: in progress...")
    for k in range(dts_size):
        progressbar(k / (dts_size - 1))
        save(f"{EXPORT_DATA_PATH}/{files_names[k]}_{w}.png", rcp[k])
    print()
    print("Log: Sauvegarde des pathces: done!")


# Permet de transformer une image grise de shape (w,l) en (w,l,d)
def imagegrey_to_3dgreyimage(img):
    if len(img.shape) > 3:
        return img
    ish = img.shape
    res = zeros((ish[0], ish[1], 3))
    for line in range(ish[0]):
        for row in range(ish[1]):
            v = img[line, row]
            res[line, row] = asarray([v, v, v])
    return res


# Permet de transformer une liste de liste de patch (w,l,d) en liste de patch flat
def compact_patches(list_patch):
    return [asarray(p).flatten() for patches in list_patch for p in patches]
