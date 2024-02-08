import sys
sys.path.append("../")

from env.env_model import build_env
from HMARL.utils.options import parse_options
from HMARL.agent import build_agent
import os
import cv2
from tqdm import tqdm
import torch
from os import path as osp
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--input', default='./input', type=str, help='test image path')
parser.add_argument('--output', default='./output', type=str, help='output image path')
parser.add_argument('--scale', default=4, type=int, help='set upsample scale')
parser.add_argument('--device', default='cuda', type=str, help='GPU or CPU')
parser.add_argument('--opt', default="../experiments_log/hmarl_experiments/hmarl_version_0", type=str, help='opt')
args = parser.parse_args()
_, opt = parse_options(test=args.opt)

opt["device"] = args.device
env = build_env(opt)
agent = build_agent(opt)
checkpoint_path = osp.join(opt["path"]["checkpoint"], "best_model.pth")
checkpoint = torch.load(checkpoint_path, map_location=opt["device"])["state_dict"]
agent.load_checkpoint(checkpoint)
agent.ppo.eval()

scale = args.scale
lr_path = args.input
save_path = args.output
scale_lise = [[2], [2, 1.5], [2, 2]]

file_list = os.listdir(lr_path)

for file_name in tqdm(file_list):
    state = cv2.imread(os.path.join(lr_path, file_name))
    for s in scale_lise[scale - 2]:
        state_ = env.reset_val(state, s)
        action = agent.select_action_play(state_)
        next_state = env.step_val(action)
        state = next_state
    cv2.imwrite(osp.join(save_path, file_name), state)

