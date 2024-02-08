import cv2
import numpy as np
from PIL import Image
import logging
from HMARL.utils.logger import get_root_logger
from HMARL.env.operations import build_operation
from HMARL.env.common import UpSample
from HMARL.env.reward.compute_reward import ComputeReward
import HMARL.utils.utils_image as util


class SuperResolutionEnv:

    def __init__(self, opt):
        self.opt = opt
        self.scale = opt["scale"]
        # batch size
        self.batch_size = opt["train"]["batch_size"]

        # train_state
        self.hr_patch = None
        self.lr_patch = None

        # demo_state
        self.state_rgb = np.array([])

        # val_state
        self.state_lr = np.array([])
        self.state_hr = np.array([])
        self.hr_bgr = np.array([])

        # Create operation
        self.operation = build_operation(opt)

        # Upsampling method
        self.upsampling = UpSample("bicubic", opt["device"]).to(opt["device"])

        # Reward function
        self.compute_reward = ComputeReward(opt)

        self.w, self.h = 0, 0

        # log
        self.logger = get_root_logger(logger_name='basicsr', log_level=logging.INFO)
        self.logger.info('Environment initialized successfully')

    # Training environment
    def reset(self, lr, hr):
        self.lr_patch, self.hr_patch = lr.numpy(), hr.numpy()
        return self.lr_patch[:, 0, :, :][:, None, :, :]

    def step(self, action, gamma_noise):
        """action size = (5, h, w)"""
        adjust_result, adjust_result_gamma = self.operation(self.lr_patch, action, only_l=True)

        adjust_result_ = np.concatenate((adjust_result, adjust_result_gamma), axis=1)
        reward = self.compute_reward(adjust_result_, self.hr_patch[:, 0][:, None, :, :],
                                     self.lr_patch[:, 0][:, None, :, :], action)
        next_state = np.concatenate(
            (adjust_result_gamma, np.clip(self.hr_patch[:, 0][:, None, :, :] + gamma_noise[:, None, :, :], 0, 255)), axis=1)
        return next_state, reward

    def reset_val(self, img_lr, scale):
        img_lr = img_lr[0:img_lr.shape[0] - img_lr.shape[0] % 4, 0:img_lr.shape[1] - img_lr.shape[1] % 4]
        img_lr = Image.fromarray(cv2.cvtColor(img_lr, cv2.COLOR_BGR2RGB), mode="RGB")
        img_lr = img_lr.resize((int(scale * img_lr.size[0]), int(scale * img_lr.size[1])), Image.BICUBIC)
        img_lr = np.clip(np.array(img_lr), 0, 255)
        img_lr_y = cv2.cvtColor(img_lr, cv2.COLOR_RGB2LAB)
        self.state_lr = util.single2tensor3(img_lr_y)[None, :, :, ]

        self.w, self.h, _ = img_lr.shape
        w_pad = 8 - img_lr.shape[0] % 8 if img_lr.shape[0] % 8 != 0 else 0
        h_pad = 8 - img_lr.shape[1] % 8 if img_lr.shape[1] % 8 != 0 else 0
        self.state_lr = np.pad(self.state_lr, ((0, 0), (0, 0), (0, w_pad), (0, h_pad)), 'constant')

        return self.state_lr[:, 0][:, None, :, :]

    def step_val(self, action):
        adjust_result, adjust_result_gamma = self.operation.forward_only_image(self.state_lr, action, only_l=False)

        adjust_result = adjust_result[:, :, 0: self.w, 0: self.h]
        self.state_lr = self.state_lr[:, :, 0: self.w, 0: self.h]

        # next state
        next_state = cv2.cvtColor(adjust_result[0].transpose(1, 2, 0).astype(np.uint8), cv2.COLOR_LAB2BGR)

        return next_state

    def render(self):
        pass
