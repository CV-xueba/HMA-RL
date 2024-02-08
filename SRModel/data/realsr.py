import cv2
import os
import logging
from SRModel.utils.logger import get_root_logger
import SRModel.utils.utils_image as util
from torch.utils.data import Dataset


class RealSR(Dataset):
    def __init__(self, opt):
        self.opt = opt
        self.train = False
        self.split = "test"
        self.num_worker = opt["datasets"]["train"]["num_worker"]

        # Setting the data root directory
        self._set_filesystem(opt["datasets"]["val"]["dataroot"])

        # log
        self.logger = get_root_logger(logger_name='basicsr', log_level=logging.INFO)

        # Initialize paths
        self.images_lr, self.images_sr = self._scan()

    def _load_file(self, idx):
        #  Load path
        sr = self.images_sr[idx]
        lr = self.images_lr[idx]

        # Load validation image
        _, filename = os.path.split(sr)
        sr = cv2.imread(sr)
        lr = cv2.imread(lr)

        return lr, sr, filename

    def __getitem__(self, idx):
        img_lr, img_sr, _ = self._load_file(idx)
        img_lr, img_sr = util.img2tensor([img_lr, img_sr], bgr2rgb=True, float32=True)
        return img_lr / 255., img_sr / 255.

    def __len__(self):
        return len(self.images_sr)

    def _set_filesystem(self, dir_data):
        self.apath = dir_data
        self.dir_sr = os.path.join(self.apath, 'sr')
        self.dir_lr = os.path.join(self.apath, 'lr')
        self.ext = '.png'

    def _scan(self):
        sr_file_list = os.listdir(self.dir_sr)
        lr_file_list = os.listdir(self.dir_lr)
        sr_file_list.sort()
        lr_file_list.sort()
        sr_list = [os.path.join(self.dir_sr, sr_file) for sr_file in sr_file_list]
        lr_list = [os.path.join(self.dir_lr, lr_file) for lr_file in lr_file_list]
        return lr_list, sr_list
