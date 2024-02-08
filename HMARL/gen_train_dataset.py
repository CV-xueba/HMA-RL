from one_stage_hmarl import OneStage
from os import path as osp
import os
import cv2
from tqdm import tqdm


# Load configuration and set random seed
one_stage = OneStage(test_exp="../experiments_log/hmarl_experiments/hmarl_version_0")

scale = [2, 3, 4]
scale_list = [[2, 1], [2, 1, 1.5, 1], [2, 1, 2, 1]]
lr_path_root = "../datasets/DIV2K/trian/lr/"
save_path_root = "../datasets/div2k_one_stage/lr/"

for s in scale:
    lr_path = os.path.join(lr_path_root, f"x{scale[s]}")
    file_list = os.listdir(lr_path)
    for file_name in tqdm(file_list):
        one_stage_output = one_stage.hmarl_run(os.path.join(lr_path, file_name), scale=s, scale_list=scale_list)
        cv2.imwrite(osp.join(save_path_root, f"x{scale[s]}", file_name), one_stage_output)

