import matplotlib.pyplot as plt
from Tools.mytoolsbox import path_file_to_greyset, calc_dts_sift, afficher_n_images_keypoints
from Tools.setup import DATA_PATH

noi = 100  # number of images min 100 max 1000

print("Log: Chargement des images: in progress...")
dataset, classes, files_name = path_file_to_greyset(DATA_PATH)
print("Log: Chargement des images: done!")

print("Log: Calcule des SIFT: in progress...")
dataset_sift = calc_dts_sift([dataset[0][:noi], dataset[1][:noi]])
print()
print("Log: Calcule des SIFT: done!")

print("Log: Affichage des keypoints: in progress...")
afficher_n_images_keypoints(5, dataset[0][:noi], dataset_sift[0])
plt.show()
