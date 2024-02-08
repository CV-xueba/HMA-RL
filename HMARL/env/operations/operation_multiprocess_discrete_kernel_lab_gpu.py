import numpy as np
import torch
import torch.nn as nn


class AdaptiveBilateralFiltering:
    def __init__(self, opt):
        self.opt = opt
        self.process_num = opt["env"]["operation"]["process_num"]
        self.radius = opt["env"]["kernel_radius"]
        device_id = 2
        self.device = f"cuda:{device_id}"
        self.batch_size = 32

        self.img_l = np.array([])
        self.img_lab = np.array([])
        self.img_padding_l = np.array([])
        self.img_padding_lab = np.array([])
        self.out_filter = np.array([])
        self.sigma_s = np.array([])
        self.sigma_r = np.array([])
        self.bias = np.array([])

    def get_gaussian_kernel(self, sigma_s):
        b, wh, c = sigma_s.shape
        x = y = np.linspace(-self.radius, self.radius, int(2 * self.radius + 1))
        x, y = np.meshgrid(x, y)
        x, y = torch.from_numpy(x.reshape(1, 1, -1)).double().to(self.device), torch.from_numpy(y.reshape(1, 1, -1)).double().to(self.device)
        mat = torch.ones((b, wh, (2 * self.radius + 1)**2)).double().to(self.device)
        x_mat, y_mat = mat * x, mat * y
        arg = -(x_mat ** 2 + y_mat ** 2) / (2 * sigma_s ** 2 + 1e-8)
        h = torch.exp(arg)
        h = h / torch.sum(h, dim=2, keepdim=True)
        return h

    def filtering(self, img_lab, parameter_map):
        b, c, h, w = img_lab.shape
        batch_size = self.batch_size
        outFilter = np.zeros((b, c, h, w)).astype(np.float64)
        img_padding_lab = torch.from_numpy(np.pad(img_lab, ((0, 0), (0, 0), (self.radius, self.radius),
                                                            (self.radius, self.radius)), 'constant')).double().to(self.device)
        img_l = torch.from_numpy(img_lab[:, 0, :, :].reshape(b, -1, 1)).double().to(self.device)
        parameter_map = torch.from_numpy(parameter_map).double().to(self.device)
        sigma_s = parameter_map[:, 0].reshape(b, -1, 1)
        sigma_r = parameter_map[:, 1].reshape(b, -1, 1)
        bias = parameter_map[:, 2].reshape(b, -1, 1)

        for i in range(0, b, batch_size):
            img_padding_l = self.array2unfold(img_padding_lab[i:np.min((b, i+batch_size)), 0, :, :][:, None, :, :])
            img_padding_a = self.array2unfold(img_padding_lab[i:np.min((b, i+batch_size)), 1, :, :][:, None, :, :])
            img_padding_b = self.array2unfold(img_padding_lab[i:np.min((b, i+batch_size)), 2, :, :][:, None, :, :])
            gaussian_kernel = self.get_gaussian_kernel(sigma_s[i:np.min((b, i+batch_size))])
            r_arg = (img_padding_l - img_l[i:np.min((b, i+batch_size))] - bias[i:np.min((b, i+batch_size))]) ** 2
            histogram_kernel = torch.exp(-r_arg / (2 * sigma_r[i:np.min((b, i+batch_size))] ** 2 + 1e-8))
            K = torch.matmul(gaussian_kernel[:, :, None, :], histogram_kernel[:, :, :, None]).squeeze(-1)
            compound_kernel = gaussian_kernel * histogram_kernel / K
            filter_result_l = torch.matmul(compound_kernel[:, :, None, :], img_padding_l[:, :, :, None]).squeeze((2, 3))
            filter_result_a = torch.matmul(compound_kernel[:, :, None, :], img_padding_a[:, :, :, None]).squeeze((2, 3))
            filter_result_b = torch.matmul(compound_kernel[:, :, None, :], img_padding_b[:, :, :, None]).squeeze((2, 3))
            outFilter[i:np.min((b, i+batch_size))] = torch.stack([filter_result_l, filter_result_a, filter_result_b], dim=1).reshape((np.min((b - i, batch_size)), c, h, w)).cpu().numpy()

        return outFilter

    def filtering_only_image(self, img_lab, parameter_map):
        b, c, h, w = img_lab.shape
        batch_size = self.batch_size
        outFilter = np.zeros((b, c, h, w)).astype(np.float64)
        img_padding_lab = torch.from_numpy(np.pad(img_lab, ((0, 0), (0, 0), (self.radius, self.radius),
                                                            (self.radius, self.radius)), 'constant')).double().to(self.device)
        img_l_all = torch.from_numpy(img_lab[:, 0, :, :]).double().to(self.device)
        parameter_map = torch.from_numpy(parameter_map).double().to(self.device)
        sigma_s_all = parameter_map[:, 0]
        sigma_r_all = parameter_map[:, 1]
        bias_all = parameter_map[:, 2]

        for i in range(0, h, batch_size):
            img_padding_l = self.array2unfold(img_padding_lab[:, 0, i:np.min((h+2*self.radius, i+batch_size+2*self.radius)), :][:, None, :, :])
            img_padding_a = self.array2unfold(img_padding_lab[:, 1, i:np.min((h+2*self.radius, i+batch_size+2*self.radius)), :][:, None, :, :])
            img_padding_b = self.array2unfold(img_padding_lab[:, 2, i:np.min((h+2*self.radius, i+batch_size+2*self.radius)), :][:, None, :, :])
            sigma_s = sigma_s_all[:, i:np.min((h, i+batch_size)), :].reshape(b, -1, 1)
            sigma_r = sigma_r_all[:, i:np.min((h, i+batch_size)), :].reshape(b, -1, 1)
            bias = bias_all[:, i:np.min((h, i+batch_size)), :].reshape(b, -1, 1)
            img_l = img_l_all[:, i:np.min((h, i+batch_size)), :].reshape(b, -1, 1)

            gaussian_kernel = self.get_gaussian_kernel(sigma_s)
            r_arg = (img_padding_l - img_l - bias) ** 2
            histogram_kernel = torch.exp(-r_arg / (2 * sigma_r ** 2 + 1e-8))
            K = torch.matmul(gaussian_kernel[:, :, None, :], histogram_kernel[:, :, :, None]).squeeze(-1)
            compound_kernel = gaussian_kernel * histogram_kernel / K
            filter_result_l = torch.matmul(compound_kernel[:, :, None, :], img_padding_l[:, :, :, None])[:,:,0,0]
            filter_result_a = torch.matmul(compound_kernel[:, :, None, :], img_padding_a[:, :, :, None])[:,:,0,0]
            filter_result_b = torch.matmul(compound_kernel[:, :, None, :], img_padding_b[:, :, :, None])[:,:,0,0]
            outFilter[:, :, i:np.min((h, i+batch_size)), :] = torch.stack([filter_result_l, filter_result_a, filter_result_b], dim=1).reshape((1, c, np.min((h-i, batch_size)), w)).cpu().numpy()

        return outFilter


    def forward_only_image(self, img_lab, action, only_l=True):
        # Convert to actual action
        filter_action = action[:, :3]
        gamma_action = action[:, 3]
        outFilter = self.filtering_only_image(img_lab, parameter_map=filter_action)
        outFilter_gamma = outFilter.copy()
        outFilter_gamma[:, 0, :, :] = outFilter_gamma[:, 0, :, :] + gamma_action

        if only_l:
            outFilter = outFilter[:, 0, :, :][:, None, :, :]
            outFilter_gamma = outFilter_gamma[:, 0, :, :][:, None, :, :]

        outFilter = np.clip(outFilter, 0, 255)
        outFilter_gamma = np.clip(outFilter_gamma, 0, 255)

        return outFilter, outFilter_gamma

    def __call__(self, img_lab, action, only_l=True):
        #  Index action to actual action
        filter_action = action[:, :3]
        gamma_action = action[:, 3]
        outFilter = self.filtering(img_lab, parameter_map=filter_action)
        outFilter_gamma = outFilter.copy()
        outFilter_gamma[:, 0, :, :] = outFilter_gamma[:, 0, :, :] + gamma_action

        if only_l:
            outFilter = outFilter[:, 0, :, :][:, None, :, :]
            outFilter_gamma = outFilter_gamma[:, 0, :, :][:, None, :, :]

        outFilter = np.clip(outFilter, 0, 255)
        outFilter_gamma = np.clip(outFilter_gamma, 0, 255)

        return outFilter, outFilter_gamma

    def array2unfold(self, tensor):
        unfold = nn.Unfold(kernel_size=2 * self.radius + 1)
        output = unfold(tensor)
        return output.permute((0, 2, 1))
