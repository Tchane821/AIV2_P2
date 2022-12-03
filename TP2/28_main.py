from sklearn.linear_model import LogisticRegression
from Tools.mytoolsbox import path_file_to_greyset, training_process_and_eval, afficher_confmat, calc_dts_vladpca
from Tools.setup import DATA_PATH, VOCAB_PATH

# Start script
rand_seed = 42
noi = 100  # number of images min 100 max 1000
path_to_voc = f"{VOCAB_PATH}/vocabulary_25.npy"

print("Log: Chargement des images: in progress...")
dataset, classes, files_name = path_file_to_greyset(DATA_PATH)
print("Log: Chargement des images: done!")

print("Log: Calcule des descripteur: in progress... ")
data_from_images = calc_dts_vladpca([dataset[0][:noi], dataset[1][:noi]], path_to_voc)
print()
print("Log: Calcule des descripteur: done! ")

print("Log: Aprentissage du modele: in progress...")
classifier = LogisticRegression()
accuracy, matrice_confusion = training_process_and_eval(classifier, data_from_images, rs=rand_seed)
print("Log: Aprentissage du modele: done!")

print("Methode de classification : LogisticRegression")
print(f"Accuracy : {accuracy}")
print(f"Matrice de confusion:\n{matrice_confusion}")

afficher_confmat(matrice_confusion, classes, "../01_export/conf_mat_ex28")

# Acc : 0685
# [[25  1  0  0  0  0  0  0  0  0]
#  [ 0 19  0  0  1  0  0  0  0  0]
#  [ 0  0 11  1  0  1  2  0  0  1]
#  [ 0  0  0 13  0  0  0  0  1  0]
#  [ 0  0  0  3 10  2  3  5  0  0]
#  [ 0  0  0  4  1 18  0  0  0  0]
#  [ 1  0  1  0  3  0  7  4  1  3]
#  [ 0  1  2  0  7  0  1 18  0  0]
#  [ 1  0  1  3  0  0  1  0  7  2]
#  [ 2  0  0  1  0  0  0  0  2  9]]