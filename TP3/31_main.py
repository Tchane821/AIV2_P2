import os
from Tools.setup import DATA_PATH, EXPORT_PATCHES_PATH

from Tools.mytoolsbox import path_file_to_colorset
from Tools.patchtools import compute_and_save_patches, kmeans_save_from_patches


print("Log: Chargement des images: in progress...")
dataset, classes, _ = path_file_to_colorset(DATA_PATH)
print("Log: Chargement des images: done!")

print("Log: Extraction des patch des images: in progress...")
for number_of_patch in [10, 25, 50]:
    for size_of_patch in [6, 20, 40]:
        compute_and_save_patches(dataset[0], size_of_patch, number_of_patch)
print("Log: Extraction des patch des images: done!")

print("Log: Calcule des modèles : in progress...")
paths_patches = os.listdir(EXPORT_PATCHES_PATH)
for number_of_centroid in [8, 16, 32, 64, 128]:
    kmeans_save_from_patches(paths_patches, number_of_centroid)
print("Log: Calcule des modèles : done!")
