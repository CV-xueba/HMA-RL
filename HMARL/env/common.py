import numpy as np
import random
from scipy.signal import convolve2d
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F



mat = [[24.966, 112.0, -18.214],
       [128.553, -74.203, -93.786],
       [65.481, -37.797, 112.0]
       ]
mat_inv = np.linalg.inv(mat)


def composite_rgb(lr_org, y_channel, model):
    if model == "lab":
        # 操作结果
        lr_lab = cv2.cvtColor(lr_org, cv2.COLOR_BGR2LAB)
        lr_lab[:, :, 0] = y_channel
        next_state = cv2.cvtColor(lr_lab, cv2.COLOR_LAB2BGR)
    elif model == "ycbcr":
        lr_ycrcb = bgr2ycbcr(lr_org, only_y=False)
        lr_ycrcb[:, :, 0] = y_channel
        next_state = ycbcr2bgr(lr_ycrcb)
    else:
        next_state = None

    return next_state


def bgr2ycbcr(img_bgr, only_y=True):
    # convert
    if only_y:
        rlt = np.dot(img_bgr, [24.966, 128.553, 65.481]) / 255.0 + 16.0
    else:
        rlt = np.matmul(img_bgr, mat) / 255.0 + [16, 128, 128]
    return rlt


def ycbcr2bgr(img_ycbcr):
    # convert
    rlt = np.maximum(0, np.minimum(255, np.matmul(img_ycbcr-[16., 128., 128.], mat_inv) * 255.))
    rlt = rlt.round()
    return rlt.astype(np.uint8)


def get_HF_mask(img, threshold=0.2):
    sobel_mask = Sobel()(img)
    sobel_mask = np.clip(sobel_mask, 0, 1)
    sobel_mask[sobel_mask < threshold] = 0
    sobel_mask[sobel_mask >= threshold] = 1
    return sobel_mask


def index2radius(index_action, disperse_action_list):
    action = np.zeros_like(index_action)
    for i in range(len(disperse_action_list)):
        action[index_action == i] = disperse_action_list[i]
    return action


def get_gradient_map(img, sobel, laplace):
    if sobel and not laplace:
        gradient_map = Sobel()(img)
    elif not sobel and laplace:
        gradient_map = Laplace()(img)
    else:
        gradient_map = np.concatenate((Sobel()(img), Laplace()(img)), axis=1)
    return gradient_map


def get_patch(img_in, img_in_prob_maps, img_tar, patch_size):
    ih, iw = img_in.shape[:2]
    tp = patch_size

    iy, ix = get_top(ih, iw, img_in_prob_maps, tp)

    img_in = img_in[iy:iy + tp, ix:ix + tp, :]
    img_tar = img_tar[iy:iy + tp, ix:ix + tp, :]

    return img_in, img_tar


def get_multi_stage_patch(img_in, img_in_prob_maps, img_tar_x4, img_tar_x2, patch_size):
    ih, iw = img_tar_x4.shape[:2]
    tp_low = patch_size // 2
    tp_x2 = patch_size
    tp_x4 = patch_size * 2

    iy, ix = get_top(ih, iw, img_in_prob_maps, tp_x4)

    img_in = img_in[iy // 4:iy // 4 + tp_low, ix // 4:ix // 4 + tp_low, :]
    img_tar_x4 = img_tar_x4[iy:iy + tp_x4, ix:ix + tp_x4, :]
    img_tar_x2 = img_tar_x2[iy // 2:iy // 2 + tp_x2, ix // 2:ix // 2 + tp_x2, :]

    return img_in, img_tar_x4, img_tar_x2, iy, ix


def get_patch_random(img_in, img_tar, patch_size):
    ih, iw = img_tar.shape[:2]
    tp = patch_size

    ix = random.randrange(0, iw - tp + 1)
    iy = random.randrange(0, ih - tp + 1)

    img_in = img_in[iy:iy + tp, ix:ix + tp, :]
    img_tar = img_tar[iy:iy + tp, ix:ix + tp, :]

    return img_in, img_tar


def get_multi_stage_patch_random(img_in, img_tar_x4, img_tar_x2, patch_size):
    ih, iw =  img_tar_x4.shape[:2]
    tp_low = patch_size // 2
    tp_x2 = patch_size
    tp_x4 = patch_size * 2

    ix = random.randrange(0, iw - tp_x4 + 1)
    iy = random.randrange(0, ih - tp_x4 + 1)

    ix, iy = ix - ix % 4, iy - iy % 4

    img_in = img_in[iy // 4:iy // 4 + tp_low, ix // 4:ix // 4 + tp_low, :]
    img_tar_x4 = img_tar_x4[iy:iy + tp_x4, ix:ix + tp_x4, :]
    img_tar_x2 = img_tar_x2[iy // 2:iy // 2 + tp_x2, ix // 2:ix // 2 + tp_x2, :]

    return img_in, img_tar_x4, img_tar_x2


def get_top(ih, iw, prob_maps, size):
    center = np.random.choice(a=len(prob_maps), p=prob_maps)
    row, col = int(center / iw), center % iw
    top, left = min(max(0, row - size // 2), ih - size), min(max(0, col - size // 2), iw - size)
    return top-top % 4, left - left % 4


def get_tensor_center(tensor):
    _, _, w, h = tensor.shape
    top_left = w // 2 - w // 4
    tensor_center = tensor[:, :, top_left: top_left + w // 2, top_left: top_left + w // 2]
    return tensor_center


def set_channel(l, channel_model="lab"):
    def _set_channel(img):
        if channel_model == "lab":
            img = np.expand_dims(cv2.cvtColor(img, cv2.COLOR_RGB2LAB)[:, :, 0], 2)
        elif channel_model == "ycbcr":
            img = np.expand_dims(bgr2ycbcr(img), 2)
        return img

    return [_set_channel(_l) for _l in l]


def to_tensor(l):
    def _np2Tensor(img):
        img = img.astype(np.float32)
        np_transpose = np.ascontiguousarray(img.transpose((2, 0, 1)))
        return np_transpose

    return [_np2Tensor(_l) for _l in l]


def augment(l, hflip=True, rot=True):
    hflip = hflip and random.random() < 0.5
    vflip = rot and random.random() < 0.5
    rot90 = rot and random.random() < 0.5

    def _augment(img):
        if hflip: img = img[:, ::-1, :]
        if vflip: img = img[::-1, :, :]
        if rot90: img = img.transpose(1, 0, 2)

        return img

    return [_augment(_l) for _l in l]


def pad_edges(im, edge):
    """Replace image boundaries with 0 without changing the size"""
    zero_padded = np.zeros_like(im)
    zero_padded[edge:-edge, edge:-edge] = im[edge:-edge, edge:-edge]
    return zero_padded


def clip_extreme(im, percent):
    """Zeroize values below the a threshold and clip all those above"""
    # Sort the image
    im_sorted = np.sort(im.flatten())
    # Choose a pivot index that holds the min value to be clipped
    pivot = int(percent * len(im_sorted))
    v_min = im_sorted[pivot]
    # max value will be the next value in the sorted array. if it is equal to the min, a threshold will be added
    v_max = im_sorted[pivot + 1] if im_sorted[pivot + 1] > v_min else v_min + 10e-6
    # Clip an zeroize all the lower values
    return np.clip(im, v_min, v_max) - v_min


def create_gradient_map(im, window=5, percent=.97):
    """Create a gradient map of the image blurred with a rect of size window and clips extreme values"""
    # Calculate gradients
    gx, gy = np.gradient(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY))
    # Calculate gradient magnitude
    gmag, gx, gy = np.sqrt(gx ** 2 + gy ** 2), np.abs(gx), np.abs(gy)
    # Pad edges to avoid artifacts in the edge of the image
    gx_pad, gy_pad, gmag = pad_edges(gx, int(window)), pad_edges(gy, int(window)), pad_edges(gmag, int(window))
    lm_x, lm_y, lm_gmag = clip_extreme(gx_pad, percent), clip_extreme(gy_pad, percent), clip_extreme(gmag, percent)
    # Sum both gradient maps
    grads_comb = lm_x / lm_x.sum() + lm_y / lm_y.sum() + gmag / gmag.sum()
    # Blur the gradients and normalize to original values
    loss_map = convolve2d(grads_comb, np.ones(shape=(window, window)), 'same') / (window ** 2)
    # Normalizing: sum of map = numel
    return loss_map / np.mean(loss_map)


def create_probability_map(loss_map, crop):
    """Create a vector of probabilities corresponding to the loss map"""
    # Blur the gradients to get the sum of gradients in the crop
    blurred = convolve2d(loss_map, np.ones([crop // 2, crop // 2]), 'same') / ((crop // 2) ** 2)
    # Zero pad s.t. probabilities are NNZ only in valid crop centers
    prob_map = pad_edges(blurred, crop // 2)
    # Normalize to sum to 1
    prob_vec = prob_map.flatten() / prob_map.sum() if prob_map.sum() != 0 else np.ones_like(prob_map.flatten()) / \
                                                                               prob_map.flatten().shape[0]
    return prob_vec


def _get_sobel_kernel_3x3() -> torch.Tensor:
    """Utility function that returns a sobel kernel of 3x3"""
    return torch.tensor([
        [-1., 0., 1.],
        [-2., 0., 2.],
        [-1., 0., 1.],
    ])


def _get_laplace_kernel_3x3() -> torch.Tensor:
    """Utility function that returns a sobel kernel of 3x3"""
    return torch.tensor([
        [1.,  1., 1.],
        [1., -8., 1.],
        [1.,  1., 1.]
    ])

def _get_laplace_gaussian_kernel_3x3() -> torch.Tensor:
    """Utility function that returns a sobel kernel of 3x3"""
    return torch.tensor([
        [0, 0, 1, 0, 0],
        [0, 1, 2, 1, 0],
        [1, 2, -16, 2, 1],
        [0, 1, 2, 1, 0],
        [0, 0, 1, 0, 0]
    ])


class Sobel(nn.Module):
    r"""Computes the first order image derivative in both x and y using a Sobel
    operator.

    Return:
        torch.Tensor: the sobel edges of the input feature map.

    Shape:
        - Input: :math:`(B, C, H, W)`
        - Output: :math:`(B, C, 2, H, W)`

    """

    def __init__(self) -> None:
        super(Sobel, self).__init__()
        self.kernel_x, self.kernel_y = self.get_sobel_kernel()
        self.pad = nn.ReplicationPad2d(1)

    @staticmethod
    def get_sobel_kernel():
        kernel_x = _get_sobel_kernel_3x3()
        kernel_y = kernel_x.transpose(0, 1)
        return [kernel_x, kernel_y]

    def forward(self, input_data):
        input_type_is_tensor = torch.is_tensor(input_data)
        if not input_type_is_tensor:
            input_data = torch.from_numpy(input_data)

        # prepare kernel
        kernel_x: torch.Tensor = self.kernel_x.to(input_data).to(input_data)
        kernel_y: torch.Tensor = self.kernel_y.to(input_data).to(input_data)
        kernel_x = kernel_x.repeat(1, 1, 1, 1)
        kernel_y = kernel_y.repeat(1, 1, 1, 1)
        # convolve input tensor with sobel kernel
        input_data = self.pad(input_data)
        gx = F.conv2d(input_data, kernel_x, padding=0)
        gy = F.conv2d(input_data, kernel_y, padding=0)

        gradient = torch.sqrt(gx**2 + gy**2)

        if not input_type_is_tensor:
            gradient = gradient.cpu().numpy()

        return gradient


class Laplace(nn.Module):
    r"""Computes the first order image derivative in both x and y using a Sobel
    operator.

    Return:
        torch.Tensor: the sobel edges of the input feature map.

    Shape:
        - Input: :math:`(B, C, H, W)`

    """

    def __init__(self) -> None:
        super(Laplace, self).__init__()
        self.kernel = self.get_laplace_kernel()
        self.pad = nn.ReplicationPad2d(self.kernel.shape[0]//2)

    @staticmethod
    def get_laplace_kernel():
        kernel = _get_laplace_kernel_3x3()
        return kernel

    def forward(self, input_data):
        input_type_is_tensor = torch.is_tensor(input_data)
        if not input_type_is_tensor:
            input_data = torch.from_numpy(input_data)

        # prepare kernel
        kernel: torch.Tensor = self.kernel.to(input_data).to(input_data)
        kernel = kernel.repeat(1, 1, 1, 1)
        # convolve input tensor with sobel kernel
        input_data = self.pad(input_data)
        gradient = F.conv2d(input_data, kernel, padding=0)
        if not input_type_is_tensor:
            gradient = gradient.cpu().numpy()

        return gradient


class UpSample(nn.Module):
    def __init__(self, mode, device):
        super(UpSample, self).__init__()
        self.mode = mode
        self.device = device

    def forward(self, x, scale):
        x = torch.from_numpy(x).float().to(self.device)
        shape = x.shape
        if len(shape) == 3:
            x = x.permute(2, 0, 1).unsqueeze(0)
        x = F.interpolate(x, scale_factor=scale, mode=self.mode)
        if len(shape) == 3:
            x = x.squeeze(0).permute(1, 2, 0)
        return x.cpu().numpy()
