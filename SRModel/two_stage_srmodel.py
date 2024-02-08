from SRModel.utils.options import parse_options
from SRModel.models import build_model
import torch
from os import path as osp
import cv2
import SRModel.utils.utils_image as util
import numpy as np


class TwoStage:
    def __init__(self, test_exp="srmodel_experiments"):
        args, opt = parse_options(test=test_exp)
        self.device = "cuda"
        opt["path"]["test_results"] = osp.join(opt["path"]["experiments_root"], "test_results")
        self.model = build_model(opt)

        checkpoint_path_g = osp.join(opt["path"]["checkpoint"], f"best_model.pth")
        checkpoint_path_d = osp.join(opt["path"]["checkpoint"], f"best_model_D.pth")
        checkpoint_g = torch.load(checkpoint_path_g, map_location=self.device)["state_dict"]
        checkpoint_d = torch.load(checkpoint_path_d, map_location=self.device)["state_dict"]
        self.model.load_checkpoint(checkpoint_g, checkpoint_d)

    def sr_run(self, one_stage_img):
        input_ = util.img2tensor([one_stage_img], bgr2rgb=True, float32=True)[0].unsqueeze(0).to(self.device) / 255.
        input_ = input_[:, :, 0: one_stage_img.shape[0] - one_stage_img.shape[0] % 32, 0: one_stage_img.shape[1] - one_stage_img.shape[1] % 32]

        self.model.net_g.eval()
        with torch.no_grad():
            output = self.model.net_g(input_)

        img_sr = np.clip(output.detach().cpu()[0].numpy().transpose(1, 2, 0), 0, 1)
        img_sr = cv2.cvtColor(img_sr, cv2.COLOR_RGB2BGR)

        return img_sr * 255

