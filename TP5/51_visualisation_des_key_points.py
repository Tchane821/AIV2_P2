import os

from setup import DATA_VIDEO_2_PATH, DATA_VIDEO_2_KEY_PATH
from stip import read_stip_file

video_paths = os.listdir(DATA_VIDEO_2_PATH)
dotkeys_paths = os.listdir(DATA_VIDEO_2_KEY_PATH)

kp, desc = read_stip_file(f"{DATA_VIDEO_2_KEY_PATH}/{dotkeys_paths[0]}")

desc_hog = desc[:, :72]
desc_hof = desc[:, 72:]

print(desc_hog.shape)
print(desc_hof.shape)
