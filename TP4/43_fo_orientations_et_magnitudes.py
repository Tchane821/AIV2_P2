import os

from setup import DATA_VIDEO_1_PATH
from video_tools import calc_opticalflux, calc_fo_2_tuple

listed_video_path = os.listdir(DATA_VIDEO_1_PATH)
k = 0

print("Log: Calcule des flux optique: in progress...")
fo = calc_opticalflux(listed_video_path[k])
print(f"\tShape du flux optique: {fo.shape}")
print("Log: Calcule des flux optique: done!")

print("Log: Conversion des flux optique au format OM: in progress...")
fo_om = calc_fo_2_tuple(fo)
print(f"\tShape du flux optique OM: {fo_om.shape}")
print("Log: Conversion des flux optique au format OM: done!")
