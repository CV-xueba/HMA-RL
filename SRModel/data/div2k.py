import cv2
import os
import SRModel.data.common as common
import numpy as np
import logging
from SRModel.utils.logger import get_root_logger
import SRModel.utils.utils_image as util
import random
from torch.utils.data import Dataset


class DIV2K(Dataset):
    def __init__(self, opt):
        self.opt = opt
        self.train = opt["is_train"]
        self.split = "train" if opt["is_train"] else "test"
        self.random_get_patch = opt["datasets"]["train"]["random_get_patch"]
        self.load_ext = opt["datasets"]["train"]["load_ext"]
        self.num_worker = opt["datasets"]["train"]["num_worker"]

        # noise
        self.noise_level = self.opt["datasets"]["train"]["noise_level"]

        # Upsampling rate, patch size
        self.patch_size = opt["datasets"]["train"]["patch_size"]

        # Setting the data root directory
        self._set_filesystem(opt["datasets"]["train"]["dataroot"])

        # log
        self.logger = get_root_logger(logger_name='basicsr', log_level=logging.INFO)

        self.images_lr = []
        self.images_hr = []
        self.images_prob_maps = []
        self.init_data()

    def init_data(self):
        def _load_bin():
            self.logger.info("Loading training data")
            self.images_hr = np.load(self._name_hrbin(), allow_pickle=True)
            self.logger.info("Successfully loaded: " + self._name_hrbin())
            self.images_lr = np.load(self._name_lrbin(), allow_pickle=True)
            self.logger.info("Successfully loaded: " + self._name_lrbin())
            if not self.random_get_patch:
                self.images_prob_maps = np.load(self._name_prob_mapsbin(), allow_pickle=True)
                self.logger.info("Successfully loaded: " + self._name_prob_mapsbin())

        if self.load_ext == 'img' or not self.train:
            self.images_lr, self.images_hr = self._scan()
            self.images_prob_maps = [np.array([i]) for i in range(len(self.images_hr))]
        elif self.load_ext.find('bin') >= 0:
                _load_bin()
        else:
            self.logger.info('Please define data type')

    def _load_file(self, idx):
        hr = self.images_hr[idx]
        lr = self.images_lr[idx]
        prob_maps = self.images_prob_maps[idx]
        if not self.train or self.load_ext == 'img':
            _, filename = os.path.split(hr)
            hr = cv2.imread(hr)
            lr = cv2.imread(lr)
            prob_maps = np.ones(hr.shape[0]*hr.shape[1], dtype=np.float32)
            prob_maps /= prob_maps.sum()
        else:
            filename = str(idx + 1)

        filename = os.path.splitext(os.path.split(filename)[-1])[0]

        return prob_maps, lr, hr, filename

    def _load_file_random(self, idx):
        hr = self.images_hr[idx]
        lr = self.images_lr[idx]
        if not self.train or self.load_ext == 'img':
            _, filename = os.path.split(hr)
            hr = cv2.imread(hr)
            lr = cv2.imread(lr)
        else:
            filename = str(idx + 1)

        filename = os.path.splitext(os.path.split(filename)[-1])[0]

        return lr, hr, filename

    def _get_patch(self, lr, lr_prob_maps, hr):
        patch_size = self.patch_size

        lr, hr = common.get_patch(
            lr, lr_prob_maps, hr, patch_size
        )
        lr, hr = common.augment([lr, hr])
        return lr, hr

    def _get_patch_random(self, lr, hr):
        patch_size = self.patch_size
        lr, hr = common.get_patch_random(lr,  hr, patch_size)
        lr, hr = common.augment([lr, hr])
        return lr, hr

    def __getitem__(self, idx):
        if self.random_get_patch:
            img_lr, img_hr, file_name = self._load_file_random(idx)
            H, W, C = img_lr.shape

            rnd_h_H = random.randint(0, max(0, H - self.patch_size))
            rnd_w_H = random.randint(0, max(0, W - self.patch_size))
            img_lr = img_lr[rnd_h_H:rnd_h_H + self.patch_size, rnd_w_H:rnd_w_H + self.patch_size, :]
            img_hr = img_hr[rnd_h_H:rnd_h_H + self.patch_size, rnd_w_H:rnd_w_H + self.patch_size, :]

        else:
            img_hr_prob_maps, img_lr, img_hr, file_name = self._load_file(idx)
            H, W, C = img_lr.shape
            top, left = common.get_top(H, W, img_hr_prob_maps, self.patch_size)
            img_lr = img_lr[top:top + self.patch_size, left:left + self.patch_size, :]
            img_hr = img_hr[top:top + self.patch_size, left:left + self.patch_size, :]

        img_lr, img_hr = util.img2tensor([img_lr, img_hr], bgr2rgb=True, float32=True)

        return img_lr / 255., img_hr / 255.

    def __len__(self):
        return len(self.images_lr)

    def _set_filesystem(self, dir_data):
        self.apath = dir_data
        self.dir_hr = os.path.join(self.apath, 'hr')
        self.dir_lr = os.path.join(self.apath, 'lr')
        self.ext = '.png'

    def _name_hrbin(self):
        return os.path.join(
            self.apath,
            'bin',
            'train_bin_HR.npy'
        )

    def _name_lrbin(self):
        return os.path.join(
            self.apath,
            'bin',
            'train_bin_LR.npy'
        )

    def _name_prob_mapsbin(self):
        return os.path.join(
            self.apath,
            'bin',
            '{}_bin_prob_maps.npy'.format(self.split)
        )

    def _scan(self):
        list_hr = []
        list_lr = []

        idx_begin = 0
        idx_end = 800

        for i in range(idx_begin + 1, idx_end + 1):
            filename = '{:0>4}'.format(i)
            list_hr.append(os.path.join(self.dir_hr, filename + self.ext))
            list_lr.append(os.path.join(self.dir_lr, "x4", filename + "x4" + self.ext))
        return list_lr, list_hr
