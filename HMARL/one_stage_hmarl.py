from HMARL.env.env_model import build_env
from HMARL.utils.options import parse_options
from HMARL.agent import build_agent
import cv2
import torch
from os import path as osp


class OneStage:
    def __init__(self, test_exp="../experiments_log/hmarl_experiments/hmarl_version_0"):
        args, opt = parse_options(test=test_exp)
        opt["device"] = "cuda"
        self.env = build_env(opt)
        self.agent = build_agent(opt)
        checkpoint_path = osp.join(opt["path"]["checkpoint"], "best_model.pth")
        checkpoint = torch.load(checkpoint_path, map_location=opt["device"])["state_dict"]
        self.agent.load_checkpoint(checkpoint)
        self.agent.ppo.eval()

    def hmarl_run(self, img_path, scale, scale_list):
        state = cv2.imread(img_path)
        for s in scale_list[scale - 2]:
            state_ = self.env.reset_val(state, s)
            action = self.agent.select_action_play(state_)
            next_state = self.env.step_val(action)
            state = next_state
        return state
