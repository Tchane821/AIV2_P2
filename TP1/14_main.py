"""
-- 4 Classification par histogrammes de couleurs --
"""
import os
from skimage.io import imread
import random as pif
from sklearn.linear_model import LogisticRegression
from Tools.mytoolsbox import calc_desc_histogramme, training_process_and_eval, afficher_confmat
from Tools.setup import DATA_PATH

# Start script
rand_seed = 42
listed_files = os.listdir(DATA_PATH)
pif.Random(rand_seed).shuffle(listed_files)
listed_images = [imread(f"./{DATA_PATH}/{file}") for file in listed_files]
X = [calc_desc_histogramme(img) for img in listed_images]
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

afficher_confmat(matrice_confusion, n_classes, "../01_export/conf_mat_ex4")

# Questions Partie 4

# Q3:
# Le taux de reconnaissance est de 0.515,
# soit le classifier à environ 51,5% de chance de classé correctement une donnée
# Cela semble plutôt faible quand on sait que d'autre classifier font bien mieux.

# Q4:
# La matrice de confusion est la suivante :
# [[18  4  0  0  0  0  1  0  3  0]
#  [ 0 20  0  0  0  0  0  0  0  0]
#  [ 1  1 11  0  0  0  1  0  1  1]
#  [ 0  1  1  8  0  0  1  0  1  2]
#  [ 2  1  8  1  1  3  6  0  0  1]
#  [ 0  0  0  0  0 21  2  0  0  0]
#  [ 5  2  2  1  0  3  4  0  3  0]
#  [ 2  5  0  1  0  1  3 13  4  0]
#  [ 1  0  3  0  0  1  1  2  7  0]
#  [ 1  1  9  1  0  0  1  1  0  0]]
#
# La classe 'car' possède le meilleur taux de reconnaissance.
# La classe 'bonsai' na quasiment jamais était reconue.
# Le modèle a confondu beaucoup de 'watch' en 'Motorbikes'
# Le modèle a également beaucoup confondu les 'bonsai' en 'Motorbikes'.
#  Je peux déduire des informations précédentes que les descripteurs utiliser ne permets pas de différencier
#  facilement les montres des motos. Je peux l'expliquer, car les images de montre et de moto ont très souvent en grand
#  fond blanc uniforme.
