import os

from setup_env import DATA_VIDEO_PATH
from video_tools import calc_opticalflux

listed_video_path = os.listdir(DATA_VIDEO_PATH)

res = calc_opticalflux(listed_video_path[0])

print(res.shape)
