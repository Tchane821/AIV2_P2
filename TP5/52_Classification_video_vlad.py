from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

from mytoolsbox import training_process_and_eval, afficher_confmat
from setup import DATA_VIDEO_2_PATH, RAND_SEED
from video_tools import path_file_to_video_datasets_hog_hof

print("Log: Calcule du dataset avec descripteur HOG, HOF ou HOGHOF: in progress...")
dataset, classes, files_name = path_file_to_video_datasets_hog_hof(DATA_VIDEO_2_PATH, ho="HOF")
print("Log: Calcule du dataset avec descripteur HOG, HOF ou HOGHOF: done!")

print("Log: Aprentissage du modele: in progress...")
classifier = MLPClassifier(alpha=1, max_iter=800)
accuracy, matrice_confusion = training_process_and_eval(classifier, dataset, nb_classes=len(classes), rs=RAND_SEED)
print("Log: Aprentissage du modele: done!")

print("Methode de classification : MLPClassifier")
print(f"Accuracy : {accuracy}")
print(f"Matrice de confusion:\n{matrice_confusion}")

afficher_confmat(matrice_confusion, classes, "../01_export/conf_mat_ex52_hof")

# Cross val util:
# classe sklearn.model_selection.LeaveOneOut

# ---- HOG-HOF ----
# Methode de classification : MLPClassifier
# Accuracy : 0.5166666666666667
# Matrice de confusion:
# [[3 0 0 0 1 0 0 0 0 0 0 0 0]
#  [0 0 1 0 0 0 0 0 0 0 0 0 0]
#  [0 0 1 0 1 1 1 0 0 0 0 0 0]
#  [0 0 1 0 0 0 0 0 0 0 0 1 0]
#  [0 0 0 0 4 0 0 0 0 0 0 0 0]
#  [0 0 0 0 3 2 0 0 0 0 0 0 0]
#  [0 0 0 0 0 0 3 0 0 0 0 0 0]
#  [0 0 1 0 2 1 0 3 0 0 0 0 0]
#  [0 0 0 0 1 1 0 0 1 2 0 0 0]
#  [0 0 0 0 1 0 0 0 0 1 0 0 1]
#  [0 0 1 0 2 0 0 0 1 0 3 0 0]
#  [1 0 1 0 1 0 0 0 0 0 0 3 0]
#  [0 0 0 0 0 0 0 0 0 2 0 0 7]]

# ---- HOG ----
# Methode de classification : MLPClassifier
# Accuracy : 0.6333333333333333
# Matrice de confusion:
# [[3 0 0 0 1 0 0 0 0 0 0 0 0]
#  [0 1 0 0 0 0 0 0 0 0 0 0 0]
#  [0 1 0 0 2 0 0 0 0 1 0 0 0]
#  [0 0 0 1 0 0 0 0 0 0 0 0 1]
#  [0 0 0 0 2 2 0 0 0 0 0 0 0]
#  [0 0 0 0 4 1 0 0 0 0 0 0 0]
#  [0 0 0 0 0 0 3 0 0 0 0 0 0]
#  [0 0 1 0 0 1 0 3 0 0 1 1 0]
#  [0 0 0 0 1 1 0 0 1 1 1 0 0]
#  [0 0 0 0 0 0 0 0 0 3 0 0 0]
#  [0 0 0 0 0 0 0 0 0 0 7 0 0]
#  [0 0 0 0 0 0 0 0 0 0 1 5 0]
#  [0 0 1 0 0 0 0 0 0 0 0 0 8]]

# ---- HOF ----
# Methode de classification : MLPClassifier
# Accuracy : 0.5166666666666667
# Matrice de confusion:
# [[4 0 0 0 0 0 0 0 0 0 0 0 0]
#  [0 0 0 0 0 0 0 0 0 0 0 0 1]
#  [0 0 1 0 1 2 0 0 0 0 0 0 0]
#  [0 0 0 0 0 0 0 0 0 1 0 1 0]
#  [0 0 0 0 4 0 0 0 0 0 0 0 0]
#  [0 0 0 0 3 2 0 0 0 0 0 0 0]
#  [0 0 0 0 0 0 2 0 0 0 0 1 0]
#  [0 1 0 0 0 2 0 4 0 0 0 0 0]
#  [0 0 0 0 0 3 0 0 0 1 0 0 1]
#  [0 0 0 0 1 0 0 0 0 1 0 0 1]
#  [0 0 0 0 1 1 0 0 1 1 3 0 0]
#  [1 0 1 0 1 0 0 0 0 0 0 3 0]
#  [0 0 0 0 0 0 0 0 0 2 0 0 7]]
