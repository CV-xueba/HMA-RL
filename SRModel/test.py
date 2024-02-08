from utils.options import parse_options
from models import build_model
import torch
from os import path as osp
import cv2
import utils.utils_image as util
import os
import numpy as np
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--input', default='./input', type=str, help='test image path')
parser.add_argument('--output', default='./output', type=str, help='output image path')
parser.add_argument('--scale', default=4, type=int, help='set upsample scale')
parser.add_argument('--device', default='cuda', type=str, help='GPU or CPU')
parser.add_argument('--opt', default="../experiments_log/srmodel_experiments", type=str, help='opt')
args = parser.parse_args()

_, opt = parse_options(test=args.opt)

model = build_model(opt)
checkpoint_path_g = osp.join(opt["path"]["checkpoint"], f"best_model.pth")
checkpoint_path_d = osp.join(opt["path"]["checkpoint"], f"best_model_D.pth")
checkpoint_g = torch.load(checkpoint_path_g, map_location=args.device)["state_dict"]
checkpoint_d = torch.load(checkpoint_path_d, map_location=args.device)["state_dict"]
model.load_checkpoint(checkpoint_g, checkpoint_d)

lr_list = os.listdir(args.input)

for i in tqdm(range(len(lr_list))):

    lr_path = os.path.join(args.input, lr_list[i])
    _, file_name = os.path.split(lr_path)

    img_lr = cv2.imread(lr_path)

    h, w, c = img_lr.shape
    w_pad = 32 - img_lr.shape[0] % 32 if img_lr.shape[0] % 32 != 0 else 0
    h_pad = 32 - img_lr.shape[1] % 32 if img_lr.shape[1] % 32 != 0 else 0
    img_lr = np.pad(img_lr, ((0, w_pad), (0, h_pad), (0, 0)), 'constant')
    input_ = util.img2tensor([img_lr], bgr2rgb=True, float32=True)[0].unsqueeze(0).to(args.device) / 255.

    model.net_g.eval()
    with torch.no_grad():
        output = model.net_g(input_)

    img_sr = output.cpu().numpy()[0].transpose(1, 2, 0)
    img_sr = cv2.cvtColor(img_sr, cv2.COLOR_RGB2BGR)
    img_sr = img_sr[0: h, 0: w, :]
    cv2.imwrite(os.path.join(args.output, file_name), img_sr * 255)
