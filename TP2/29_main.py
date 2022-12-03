from sklearn.linear_model import LogisticRegression

from Tools.mytoolsbox import path_file_to_greyset, training_process_and_eval, afficher_confmat, calc_dts_vladgeo
from Tools.setup import DATA_PATH, VOCAB_PATH

# Start script
rand_seed = 42
noi = 1000  # number of images min 100 max 1000
n_split = 5
path_to_voc = f"{VOCAB_PATH}/vocabulary_25.npy"

print("Log: Chargement des images: in progress...")
dataset, classes, files_name = path_file_to_greyset(DATA_PATH)
print("Log: Chargement des images: done!")

print("Log: Calcule des descripteur: in progress... ")
data_from_images = calc_dts_vladgeo([dataset[0][:noi], dataset[1][:noi]], n_split, path_to_voc)
print()
print("Log: Calcule des descripteur: done! ")

print("Log: Aprentissage du modele: in progress...")
classifier = LogisticRegression()
accuracy, matrice_confusion = training_process_and_eval(classifier, data_from_images, rs=rand_seed)
print("Log: Aprentissage du modele: done!")

print("Methode de classification : LogisticRegression")
print(f"Accuracy : {accuracy}")
print(f"Matrice de confusion:\n{matrice_confusion}")

afficher_confmat(matrice_confusion, classes, "../01_export/conf_mat_ex29")
#
# Accuracy : 0.865
# Matrice de confusion:
# [[26  0  0  0  0  0  0  0  0  0]
#  [ 0 19  0  0  0  0  0  1  0  0]
#  [ 1  0 13  0  1  0  1  0  0  0]
#  [ 0  0  0 14  0  0  0  0  0  0]
#  [ 0  2  0  0 19  0  0  1  0  1]
#  [ 0  0  0  0  0 23  0  0  0  0]
#  [ 0  0  0  1  2  1 12  2  1  1]
#  [ 0  0  1  0  4  0  0 22  2  0]
#  [ 0  0  0  1  0  0  1  0 13  0]
#  [ 0  1  0  1  0  0  0  0  0 12]]
