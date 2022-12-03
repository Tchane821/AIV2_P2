from sklearn.linear_model import LogisticRegression
from Tools.mytoolsbox import path_file_to_greyset, training_process_and_eval, afficher_confmat, calc_dts_sacmotsvisuels
from Tools.setup import DATA_PATH, VOCAB_PATH

# Start script
rand_seed = 42
noi = 1000  # number of images min 100 max 1000
path_to_voc = f"{VOCAB_PATH}/vocabulary_25.npy"

print("Log: Chargement des images: in progress...")
dataset, classes, files_name = path_file_to_greyset(DATA_PATH)
print("Log: Chargement des images: done!")

print("Log: Calcule des descripteur: in progress... ")
data_from_images = calc_dts_sacmotsvisuels([dataset[0][:noi], dataset[1][:noi]], path_to_voc)
print()
print("Log: Calcule des descripteur: done! ")

print("Log: Aprentissage du modele: in progress...")
classifier = LogisticRegression()
accuracy, matrice_confusion = training_process_and_eval(classifier, data_from_images, rs=rand_seed)
print("Log: Aprentissage du modele: done!")

print("Methode de classification : LogisticRegression")
print(f"Accuracy : {accuracy}")
print(f"Matrice de confusion:\n{matrice_confusion}")

afficher_confmat(matrice_confusion, classes, "../01_export/conf_mat_ex26")

# Questions Partie 6

# Acc = 0.43 avec 25 mots
# mat conf =
# [[17  0  0  4  0  1  1  1  2  0]
#  [ 1 12  1  0  0  1  0  3  1  1]
#  [ 3  0  7  0  1  0  4  0  0  1]
#  [ 2  0  0  8  0  0  0  0  3  1]
#  [ 0  1  1  2  6  5  3  5  0  0]
#  [ 3  2  2  2  4  9  0  0  0  1]
#  [ 0  0  2  0  2  1  7  2  0  6]
#  [ 0  6  1  0  6  2  3 11  0  0]
#  [ 0  0  1  6  0  0  2  0  4  2]
#  [ 3  0  1  1  0  3  1  0  0  5]]
