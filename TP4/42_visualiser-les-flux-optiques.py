import os

import matplotlib.pyplot as plt

from setup import DATA_VIDEO_1_PATH, EXPORT_DATA_FLUXOPT
from video_tools import calc_opticalflux, conv_optical_flux_2_hsv, conv_fo_hsv_2_fo_rgb, save_video_numpyarray, \
    afficher_image_k
from numpy import load

listed_video_path = os.listdir(DATA_VIDEO_1_PATH)

for k in range(len(listed_video_path)):
    print(f"\n ----- Log: IMG {k + 1} / {len(listed_video_path)}: in progress... -----")

    print("Log: Calcule des flux optique: in progress...")
    fo = calc_opticalflux(listed_video_path[k])
    print(f"\tShape du flux optique: {fo.shape}")
    print("Log: Calcule des flux optique: done!")

    print("Log: Conversion des flux optique au format HSV: in progress...")
    fo_hsv = conv_optical_flux_2_hsv(fo)
    print(f"\tShape du flux optique hsv: {fo_hsv.shape}")
    print("Log: Conversion des flux optique au format HSV: done!")

    print("Log: Conversion de l'image HSV en RGB: in progress...")
    fo_rgb = conv_fo_hsv_2_fo_rgb(fo_hsv)
    print("Log: Conversion de l'image HSV en RGB: done!")

    print("Log: Save images numpy format: in progress...")
    save_video_numpyarray(fo_hsv, EXPORT_DATA_FLUXOPT, f"HSV_{listed_video_path[k]}")
    save_video_numpyarray(fo_rgb, EXPORT_DATA_FLUXOPT, f"RGB_{listed_video_path[k]}")
    print("Log: Save images numpy format: done!")

    print(f"\n ----- Log: IMG {k + 1} / {len(listed_video_path)}: done! -----")

# Affichage d'exemple
fo_rgb = load("../01_export/tp4_flux_optic/RGB_Diving-Side_001.avi.npy")

for k in range(15):
    afficher_image_k(fo_rgb, k)

plt.show()
