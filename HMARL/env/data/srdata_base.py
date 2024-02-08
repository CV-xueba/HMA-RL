import cv2
cv2.setNumThreads(0)
import os
import HMARL.env.common as common
import numpy as np
import logging
from HMARL.utils.logger import get_root_logger
import HMARL.utils.utils_image as util
import random
from torch.utils.data import Dataset
from PIL import Image


class SRData(Dataset):
    def __init__(self, opt):
        self.opt = opt
        self.train = opt["is_train"]
        self.split = "train" if opt["is_train"] else "test"
        self.random_get_patch = opt["datasets"]["train"]["random_get_patch"]
        self.train_image_nums = opt["datasets"]["train"]["image_nums"]
        self.load_ext = opt["datasets"]["train"]["load_ext"]
        self.num_worker = opt["datasets"]["train"]["num_worker"]

        # Upsampling factor, patch size
        self.scale = opt["scale"]
        self.patch_size = opt["datasets"]["train"]["patch_size"]
        self.lr_patch_size = opt["datasets"]["train"]["patch_size"] // self.scale

        # Setting the data root directory
        self._set_filesystem(opt["datasets"]["train"]["dataroot"])

        # log
        self.logger = get_root_logger(logger_name='basicsr', log_level=logging.INFO)

        self.images_hr = []
        self.images_prob_maps = []
        self.init_data()

    def init_data(self):
        def _load_bin():
            self.logger.info("Loading training data")
            self.images_hr = np.load(self._name_hrbin(), allow_pickle=True)
            self.logger.info("Successfully loaded: " + self._name_hrbin())
            if not self.random_get_patch:
                self.images_prob_maps = np.load(self._name_prob_mapsbin(), allow_pickle=True)
                self.logger.info("Successfully loaded: " + self._name_prob_mapsbin())

        if self.load_ext == 'img' or not self.train:
            self.images_hr = self._scan()
            self.images_prob_maps = [np.array([i]) for i in range(len(self.images_hr))]
        elif self.load_ext.find('bin') >= 0:
                _load_bin()
        else:
            self.logger.info('Please define data type')

    def _load_file(self, idx):
        hr = self.images_hr[idx]
        prob_maps = self.images_prob_maps[idx]
        if not self.train or self.load_ext == 'img':
            filename = hr
            hr = cv2.imread(hr)
            prob_maps = np.ones(hr.shape[0]*hr.shape[1], dtype=np.float32)
            prob_maps /= prob_maps.sum()
        else:
            filename = str(idx + 1)

        filename = os.path.splitext(os.path.split(filename)[-1])[0]

        return prob_maps, hr, filename

    def _load_file_random(self, idx):
        hr = self.images_hr[idx]
        if not self.train or self.load_ext == 'img':
            filename = hr
            hr = cv2.imread(hr)
        else:
            filename = str(idx + 1)

        filename = os.path.splitext(os.path.split(filename)[-1])[0]

        return hr, filename

    def __getitem__(self, idx):
        if self.random_get_patch:
            img_hr, _ = self._load_file_random(idx)
            H, W, C = img_hr.shape

            rnd_h_H = random.randint(0, max(0, H - self.patch_size))
            rnd_w_H = random.randint(0, max(0, W - self.patch_size))
            img_hr = img_hr[rnd_h_H:rnd_h_H + self.patch_size, rnd_w_H:rnd_w_H + self.patch_size, :]
        else:
            img_hr_prob_maps, img_hr, _ = self._load_file(idx)
            H, W, C = img_hr.shape
            top, left = common.get_top(H, W, img_hr_prob_maps, self.patch_size)
            img_hr = img_hr[top:top + self.patch_size, left:left + self.patch_size, :]

        img_lr = self.degradation_multi_scale_bicubic(img_hr, [1, 2, 3, 4])

        img_lr, img_hr = cv2.cvtColor(img_lr, cv2.COLOR_BGR2LAB), cv2.cvtColor(img_hr, cv2.COLOR_BGR2LAB)
        img_lr, img_hr = util.single2tensor3(img_lr), util.single2tensor3(img_hr)
        return img_lr, img_hr

    def __len__(self):
        return self.train_image_nums

    def _get_index(self, idx):
        shuffle = np.random.randint(0, self.train_image_nums)
        index = (idx + shuffle) % self.train_image_nums
        return index

    def _set_filesystem(self, dir_data):
        self.apath = dir_data
        self.dir_hr = os.path.join(self.apath, 'hr')
        self.ext = '.png'

    def _name_hrbin(self):
        return os.path.join(
            self.apath,
            'bin',
            '{}_bin_HR.npy'.format(self.split)
        )

    def _name_prob_mapsbin(self):
        return os.path.join(
            self.apath,
            'bin',
            '{}_bin_prob_maps.npy'.format(self.split)
        )

    def _scan(self):
        list_hr = []

        idx_begin = 0
        idx_end = 800

        for i in range(idx_begin + 1, idx_end + 1):
            filename = '{:0>4}'.format(i)
            list_hr.append(os.path.join(self.dir_hr, filename + self.ext))
        return list_hr

    def degradation_multi_scale_bicubic(self, img, scale_list=None):
        scale = random.choice(scale_list)
        img_blur = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), mode="RGB")
        img_blur = img_blur.resize((int(1 / scale * img_blur.size[0]), int(1 / scale * img_blur.size[1])),
                                   Image.BICUBIC)
        img_blur = img_blur.resize((int(scale * img_blur.size[0]), int(scale * img_blur.size[1])), Image.BICUBIC)
        img_blur = np.clip(np.array(img_blur), 0, 255)
        img_blur = cv2.cvtColor(img_blur, cv2.COLOR_RGB2BGR)
        return img_blur
