from collections import OrderedDict

import numpy as np

import lpips as lp

from .psnr import psnr
from .ssim import calculate_ssim as ssim
from .best_psnr import best_psnr


class IQA:
    referecnce_metrics = ["psnr", "ssim", "best_psnr", "best_ssim", "lpips"]
    nonreference_metrics = ["niqe", "piqe", "brisque"]
    supported_metrics = referecnce_metrics + nonreference_metrics

    def __init__(self, metrics, lpips_type="alex", cuda_id=0):
        for metric in self.supported_metrics:
            if not (metric in self.supported_metrics):
                raise KeyError(
                    "{} is not Supported metric. (Support only {})".format(
                        metric, self.supported_metrics
                    )
                )

        if "lpips" in metrics:
            self.lpips_fn = lp.LPIPS(net=lpips_type)
            self.cuda = True
            self.cuda_id = cuda_id
            self.lpips_fn = self.lpips_fn.to(f"cuda:{cuda_id}")
        if ("niqe" in metrics) or ("piqe" in metrics) or ("brisque" in metrics):
            import matlab.engine

            print("Starting matlab engine ...")
            self.eng = matlab.engine.start_matlab()

    def __call__(self, res, ref=None, metrics=["niqe"]):
        """
        res, ref: [0, 255]
        """
        if hasattr(self, "eng"):
            import matlab

            self.matlab_res = matlab.uint8(res.tolist())

        scores = OrderedDict()
        for metric in metrics:
            if metric in self.referecnce_metrics:
                if ref is None:
                    raise ValueError(
                        "Ground-truth refernce is needed for {}".format(metric)
                    )
                scores[metric] = getattr(self, "calculate_{}".format(metric))(res, ref)

            elif metric in self.nonreference_metrics:
                scores[metric] = getattr(self, "calculate_{}".format(metric))(res)

            else:
                raise KeyError(
                    "{} is not Supported metric. (Support only {})".format(
                        metric, self.supported_metrics
                    )
                )
        return scores

    def calculate_lpips(self, res, ref):
        if res.ndim < 3:
            return 0
        res = lp.im2tensor(res)
        ref = lp.im2tensor(ref)
        if self.cuda:
            res = res.to(f"cuda:{self.cuda_id}")
            ref = ref.to(f"cuda:{self.cuda_id}")
        score = self.lpips_fn(res, ref)
        return score.item()

    def calculate_niqe(self, res):
        return self.eng.niqe(self.matlab_res)

    def calculate_brisque(self, res):
        return self.eng.brisque(self.matlab_res)

    def calculate_piqe(self, piqe):
        return self.eng.piqe(self.matlab_res)
    
    def calculate_best_psnr(self, res, ref):
        best_psnr_, best_ssim_ = best_psnr(res, ref)
        self.best_ssim = best_ssim_
        return best_psnr_
    
    def calculate_best_ssim(self, res, ref):
        assert hasattr(self, "best_ssim")
        return self.best_ssim

    @staticmethod
    def calculate_psnr(res, ref):
        return psnr(res, ref)

    @staticmethod
    def calculate_ssim(res, ref):
        return ssim(res, ref)
