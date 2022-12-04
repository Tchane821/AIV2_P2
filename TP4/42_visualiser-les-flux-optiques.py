import os

import matplotlib.pyplot as plt

from setup import DATA_VIDEO_PATH
from video_tools import calc_opticalflux, conv_optical_flux_2_hsv, conv_fo_hsv_2_fo_rgb, save_video_numpyarray, \
    afficher_image_k

listed_video_path = os.listdir(DATA_VIDEO_PATH)

print("Log: Calcule des flux optique: in progress...")
fo = calc_opticalflux(listed_video_path[0])
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
save_video_numpyarray(fo_hsv, f"HSV_{listed_video_path[0]}")
save_video_numpyarray(fo_rgb, f"RGB_{listed_video_path[0]}")
print("Log: Save images numpy format: done!")

# from numpy import load
# fo_rgb = load("../01_export/tp4_flux_optic/RGB_Diving-Side_001.avi.npy")
#
# for k in range(15):
#     afficher_image_k(fo_rgb, k)
#
# plt.show()
