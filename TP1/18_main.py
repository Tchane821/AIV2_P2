import os
import random as pif
from skimage.io import imread
from sklearn.linear_model import LogisticRegression
from Tools.mytoolsbox import training_process_and_eval, afficher_confmat, calc_desc_geocuting
from Tools.setup_env import DATA_PATH

# Start script
rand_seed = 42
rad = 5
listed_files = os.listdir(DATA_PATH)
pif.Random(rand_seed).shuffle(listed_files)
listed_images = [imread(f"./{DATA_PATH}/{file}") for file in listed_files]
X = [calc_desc_geocuting(img, 5) for img in listed_images]
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

afficher_confmat(matrice_confusion, n_classes, "../01_export/conf_mat_ex8")

# Questions Partie 8

# Q1.
# La taille correspond au 25 histogrammes concaténer, soit un vecteur de taille 12800.

# Q5.
# Ici la précision obtenue est de 0.77 soit 77% de chance que le classifier ait raison.

# Q6.
# Nous avons une meilleure précision ici, car nous avons ajouté un details spacial au descripteur.
# La ou l'histogramme seul ne pouvait pas le représenter.
