from os import mkdir

EXPORT_GLOBAL_PATH = "../01_export"
EXPORT_MODELE_PATH = f"{EXPORT_GLOBAL_PATH}/tp3_models"
EXPORT_PATCHES_PATH = f"{EXPORT_GLOBAL_PATH}/tp3_patches"
EXPORT_SCALER_PATH = f"{EXPORT_GLOBAL_PATH}/tp3_scaler"
EXPORT_DATA_PATH = f"{EXPORT_GLOBAL_PATH}/caltech101_subset_patche"
EXPORT_DATA_HARD = f"{EXPORT_GLOBAL_PATH}/tp3_hard"
EXPORT_DATA_SOFT = f"{EXPORT_GLOBAL_PATH}/tp3_soft"
EXPORT_DATA_CONV = f"{EXPORT_GLOBAL_PATH}/tp3_conv"
EXPORT_DATA_FLUXOPT = f"{EXPORT_GLOBAL_PATH}/tp4_flux_optic"
DATA_PATH = "../02_caltech101_subset"
VOCAB_PATH = "../03_vocabulary_sift"
DATA_VIDEO_1_PATH = "../04_data_videos/video_1"
DATA_VIDEO_2_PATH = "../04_data_videos/video_2/videos"
DATA_VIDEO_2_KEY_PATH = "../04_data_videos/video_2/keypoints"
DATA_VIDEO_2_VOC_PATH = "../04_data_videos/video_2/visual_vocabularies"
RAND_SEED = 42

dirs_to_create = [
    EXPORT_GLOBAL_PATH,
    EXPORT_MODELE_PATH,
    EXPORT_PATCHES_PATH,
    EXPORT_SCALER_PATH,
    EXPORT_DATA_PATH,
    EXPORT_DATA_HARD,
    EXPORT_DATA_SOFT,
    EXPORT_DATA_CONV,
    EXPORT_DATA_FLUXOPT,
    DATA_PATH,
    VOCAB_PATH
]


def run():
    for d in dirs_to_create:
        try:
            mkdir(d)
        except FileExistsError:
            print(f"{d} existe déjà.")


if __name__ == '__main__':
    run()
