import cv2
import os
import argparse
from HMARL.one_stage_hmarl import OneStage
from SRModel.two_stage_srmodel import TwoStage

one_stage = OneStage(test_exp="experiments_log/hmarl_experiments/hmarl_version_0")
two_stage = TwoStage(test_exp="experiments_log/srmodel_experiments/srmodel_version_0")


def sr_pipline(input_path, scale, output_path):
    file_list = os.listdir(input_path)
    for file_name in file_list:
        one_stage_output = one_stage.hmarl_run(os.path.join(input_path, file_name), scale=scale, scale_list=[[2, 1], [2, 1, 1.5, 1], [2, 1, 2, 1]])
        two_stage_output = two_stage.sr_run(one_stage_output)
        cv2.imwrite(os.path.join(output_path, file_name), two_stage_output)
    print("Finished!!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='./input', type=str, help='test image path')
    parser.add_argument('--output', default='./output', type=str, help='output image path')
    parser.add_argument('--scale', default=4, type=int, help='set upsample scale')
    args = parser.parse_args()
    sr_pipline(args.input, args.scale, args.output)
