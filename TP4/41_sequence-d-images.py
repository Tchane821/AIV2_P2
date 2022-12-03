from sklearn.neural_network import MLPClassifier


from Tools.setup import DATA_VIDEO_PATH, RAND_SEED
from Tools.video_tools import path_file_to_video_datasets_histiavg
from mytoolsbox import training_process_and_eval, afficher_confmat

print("Log: calcule des descripteurs: in progress...")
dataset, classes, files_name = path_file_to_video_datasets_histiavg(DATA_VIDEO_PATH)
print()
print("Log: calcule des descripteurs: done!")

print("Log: Aprentissage du modele: in progress...")
classifier = MLPClassifier(alpha=1, max_iter=800)
accuracy, matrice_confusion = training_process_and_eval(classifier, dataset, nb_classes=len(classes), rs=RAND_SEED)
print("Log: Aprentissage du modele: done!")

print("Methode de classification : MLPClassifier")
print(f"Accuracy : {accuracy:.2f}")
print(f"Matrice de confusion:\n{matrice_confusion}")

afficher_confmat(matrice_confusion, classes, "../01_export/conf_mat_ex41")

# Accuracy : 0.77
# Matrice de confusion:
# [[0 0 0 0 0 0 0 0 0 0 0 0 0]
#  [0 0 0 1 0 0 0 0 0 0 0 0 0]
#  [0 0 1 0 1 0 0 0 0 0 0 0 0]
#  [0 0 0 2 0 0 0 0 0 0 0 0 0]
#  [0 0 0 0 0 0 0 0 0 0 0 0 0]
#  [0 0 0 0 0 0 0 0 0 0 0 0 0]
#  [0 0 0 0 0 0 1 0 0 0 0 0 0]
#  [0 0 0 0 0 0 0 2 0 0 0 0 0]
#  [0 1 0 0 0 0 0 0 1 0 0 0 0]
#  [0 0 0 0 0 0 0 0 0 0 0 0 0]
#  [0 0 0 0 0 0 0 0 0 0 1 0 0]
#  [0 0 0 0 0 0 0 0 0 0 0 2 0]
#  [0 0 0 0 0 0 0 0 0 0 0 0 0]]
