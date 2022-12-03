import os

from Tools.setup import EXPORT_DATA_PATH

from Tools.patchtools import soft_assignment
# from Tools.patchtools import hard_assignment

lf = os.listdir(EXPORT_DATA_PATH)

# hard_assignment(lf[0], 25, 128)
soft_assignment(lf[0], 25, 128)
