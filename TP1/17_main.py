import os
import random as pif
from skimage.io import imread
from sklearn.linear_model import LogisticRegression
from Tools.mytoolsbox import training_process_and_eval, afficher_confmat, calc_desc_histo_histolbp
from Tools.setup_env import DATA_PATH

# Start script
rand_seed = 42
lbp_nbpts = 8
lbp_rad = 1
listed_files = os.listdir(DATA_PATH)
pif.Random(rand_seed).shuffle(listed_files)
listed_images = [imread(f"./{DATA_PATH}/{file}") for file in listed_files]
X = [calc_desc_histo_histolbp(img, lbp_nbpts, lbp_rad) for img in listed_images]
y_name = [fname.split("_")[0] for fname in listed_files]
n_classes = list(set(y_name))
n_classes.sort()
y = [n_classes.index(n) for n in y_name]
data_from_images = [X, y]

classifier = LogisticRegression()

accuracy, matrice_confusion = training_process_and_eval(classifier, data_from_images, rs=rand_seed)

print("Methode de classification : LogisticRegression")
print(f"Accuracy : {accuracy}")
print(f"Matrice de confusion:\n{matrice_confusion}")

afficher_confmat(matrice_confusion, n_classes, "../01_export/conf_mat_ex7")

# Questions Partie 7

# Q1
# Les deux descripteurs précédents fessait 512 donc le descripteur final fera 1024.

# Q3
# La precision atteint ici 0.595 soit 59,5% de chance que le classifier ait raison.
# On note que certaine erreur perciste comme les 'hawksbill' confondu avec des 'faces' mais d'autre se sont amélioré.
# Chaque image de 'car' a été bien classifier meme si certain 'bonsai' et certain 'chandelier' se soit ajouter.

# Q4
# La reconnaissance a été améliorée, je peux en déduire qu'ajouter des descripteurs porteurs d'informations améliore,
# le processus de reconnaissance et ceux meme si les descripteurs ont des scores faible prix indépendamment.
