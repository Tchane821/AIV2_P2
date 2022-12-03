from Tools.mytoolsbox import path_file_to_colorset
from Tools.patchtools import compute_and_save_allpatches_with_out_cover
from Tools.setup import DATA_PATH

random_state = 42
noi = 1000

print("Log: Chargement des images: in progress...")
dataset, classes, files_name = path_file_to_colorset(DATA_PATH)
print("Log: Chargement des images: done!")

for w in [6, 20, 40]:
    print(f"Log: Extraction de tout les patches par image (w={w}): in progress...")
    compute_and_save_allpatches_with_out_cover([dataset[0][:noi], dataset[1][:noi]], w, files_name)
    print(f"Log: Extraction de tout les patches par image (w={w}): done!")
