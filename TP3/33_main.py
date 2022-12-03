import os
from Tools.setup import EXPORT_MODELE_PATH
from Tools.patchtools import afficher_patches_centroids

modeles_files = os.listdir(EXPORT_MODELE_PATH)
afficher_patches_centroids(modeles_files[10])
