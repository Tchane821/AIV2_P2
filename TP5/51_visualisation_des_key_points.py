import os

from setup import DATA_VIDEO_2_PATH, DATA_VIDEO_2_KEY_PATH
from stip import read_stip_file

video_paths = os.listdir(DATA_VIDEO_2_PATH)
dotkeys_paths = os.listdir(DATA_VIDEO_2_KEY_PATH)

kp_and_desc = read_stip_file(f"{DATA_VIDEO_2_KEY_PATH}/{dotkeys_paths[0]}")


