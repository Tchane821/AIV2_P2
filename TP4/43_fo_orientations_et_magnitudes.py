import os

import matplotlib.pyplot as plt
from numpy import load
from setup import DATA_VIDEO_1_PATH, EXPORT_DATA_FLUXOPT
from video_tools import calc_opticalflux, calc_fo_2_tuple, afficher_om_histo, calc_om_histo, save_video_numpyarray, \
    filtered_fo_om

k = 0
name_file_tmp = "fo_om"

if not os.path.exists(f"{EXPORT_DATA_FLUXOPT}/{name_file_tmp}.npy"):
    listed_video_path = os.listdir(DATA_VIDEO_1_PATH)

    print("Log: Calcule des flux optique: in progress...")
    fo = calc_opticalflux(listed_video_path[k])
    print(f"\tShape du flux optique: {fo.shape}")
    print("Log: Calcule des flux optique: done!")

    print("Log: Conversion des flux optique au format OM: in progress...")
    fo_om = calc_fo_2_tuple(fo)
    print(f"\tShape du flux optique OM: {fo_om.shape}")
    print("Log: Conversion des flux optique au format OM: done!")
    save_video_numpyarray(fo_om, name_file_tmp)
else:
    print("Log: Load Optical Flux au format OM: in progress...")
    fo_om = load(f"{EXPORT_DATA_FLUXOPT}/{name_file_tmp}.npy")
    print("Log: Load Optical Flux au format OM: done!")

print("Log: Calcule des histogrammes: in progress...")
histo = calc_om_histo(fo_om)
afficher_om_histo(histo)
plt.show()
print("Log: Calcule des histogrammes: done!")

print("Log: filtrage des flux optique: in progress...")
seuil_bas = 2
seuil_haut = 38
print(f"\tLog: Smin= {seuil_bas}\t Smax= {seuil_haut}")
fo_om_f = filtered_fo_om(fo_om, seuil_bas, seuil_haut)
print("Log: filtrage des flux optique: done!")
