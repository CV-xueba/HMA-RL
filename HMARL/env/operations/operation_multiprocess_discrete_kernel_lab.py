import numpy as np
import multiprocessing as mp


class AdaptiveBilateralFiltering:
    def __init__(self, opt):
        self.opt = opt
        self.process_num = opt["env"]["operation"]["process_num"]
        self.radius = opt["env"]["kernel_radius"]

        self.img_l = np.array([])
        self.img_lab = np.array([])
        self.img_padding_l = np.array([])
        self.img_padding_lab = np.array([])
        self.out_filter = np.array([])
        self.sigma_s = np.array([])
        self.sigma_r = np.array([])
        self.bias = np.array([])

    def get_gaussian_kernel(self, sigma_s):
        x = y = np.linspace(-self.radius, self.radius, int(2 * self.radius + 1))
        x, y = np.meshgrid(x, y)
        x, y = np.expand_dims(x, axis=(0, 1)), np.expand_dims(y, axis=(0, 1))
        arg = -(x ** 2 + y ** 2) / (2 * sigma_s ** 2 + 1e-8)
        h = np.exp(arg)
        h1 = h / np.sum(h, axis=(-1, -2), keepdims=True)
        return h1

    def filtering(self, img_lab, parameter_map):
        b, c, h, w = img_lab.shape

        self.img_lab = img_lab.astype(np.float64)
        self.img_padding_lab = np.pad(self.img_lab, ((0, 0), (0, 0), (self.radius, self.radius),
                                                     (self.radius, self.radius)), 'constant')

        self.img_l = img_lab.astype(np.float64)[:, 0][:, None, :, :]
        self.img_padding_l = np.pad(self.img_l, ((0, 0), (0, 0), (self.radius, self.radius),
                                                 (self.radius, self.radius)), 'constant')

        self.sigma_s = parameter_map[:, 0]
        self.sigma_r = parameter_map[:, 1]
        self.bias = parameter_map[:, 2]

        out_filter = np.zeros((b, c, h, w))
        out_filter_share = mp.Array('d', out_filter.ravel())

        process_list = [mp.Process(target=self.forward, args=(out_filter_share, start_row,))
                        for start_row in range(min(h, self.process_num))]
        [p.start() for p in process_list]
        [p.join() for p in process_list]

        outFilter = np.frombuffer(out_filter_share.get_obj(), np.double).reshape((b, c, h, w))

        return outFilter

    def __call__(self, img_lab, action, only_l=True):
        # Index action to actual action
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

    def forward(self, out_filter_share, start_row):
        b, c, h, w = self.img_lab.shape

        out_filter_share = np.frombuffer(out_filter_share.get_obj(), np.double).reshape((b, c, h, w))
        for i in range(start_row, h, self.process_num):
            for j in range(0, w):
                img_patch_lab = self.img_padding_lab[:, :, i: i + 2 * self.radius + 1, j: j + 2 * self.radius + 1]
                img_patch_lab = img_patch_lab.reshape(b, c, (2 * self.radius + 1) ** 2, 1)

                img_patch_l = self.img_padding_l[:, :, i: i + 2 * self.radius + 1, j: j + 2 * self.radius + 1]
                img_patch_l = img_patch_l.reshape(b, 1, (2 * self.radius + 1) ** 2, 1)

                r_arg = np.sum((img_patch_l - self.img_l[:, :, i, j][:, :, None, None] - self.bias[:, i, j][:, None, None, None]) ** 2, axis=1, keepdims=True)
                histogram_kernel = np.exp(-r_arg / (2 * np.expand_dims(self.sigma_r[:, i, j], axis=(1, 2, 3)) ** 2 + 1e-10))
                # Spatial domain filtering
                gaussian_kernel = self.get_gaussian_kernel(np.expand_dims(self.sigma_s[:, i, j], axis=(1, 2, 3))).reshape(b, 1, 1, (2 * self.radius + 1) ** 2)
                # Normalization
                normal = np.matmul(gaussian_kernel, histogram_kernel)
                # Bilateral filter kernel
                compound_kernel = gaussian_kernel * histogram_kernel.transpose((0, 1, 3, 2)) / normal
                # Lab channel filtering
                filter_result_lab = np.matmul(compound_kernel, img_patch_lab)
                out_filter_share[:, :, i, j] = filter_result_lab.squeeze()




