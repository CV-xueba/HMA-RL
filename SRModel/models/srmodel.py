import torch
from SRModel.models.model_base import BASEModel
from collections import OrderedDict
from torchvision.utils import make_grid
import numpy as np
import torch.nn as nn


class SRGANModel(BASEModel):
    def __init__(self, opt):
        super(SRGANModel, self).__init__(opt)
        self.use_perceptual_loss = True
        self.l1loss = nn.L1Loss()

    def optimize_parameters(self, current_iter):
        # optimize net_g
        for p in self.net_d.parameters():
            p.requires_grad = False

        self.optimizer_g.zero_grad()
        self.output = self.net_g(self.lq)

        l_g_total = 0
        loss_dict = OrderedDict()
        pixel_loss = self.l1loss(self.output, self.gt)
        loss_dict["pixel_loss"] = pixel_loss
        l_g_total += pixel_loss
        if current_iter % self.net_d_iters == 0 and current_iter > self.net_d_init_iters:
            if self.use_perceptual_loss:
                # perceptual loss
                l_g_percep, l_g_style = self.cri_perceptual(self.output, self.gt)
                if l_g_percep is not None:
                    l_g_total += l_g_percep
                    loss_dict['l_g_percep'] = l_g_percep
                if l_g_style is not None:
                    l_g_total += l_g_style
                    loss_dict['l_g_style'] = l_g_style
            # gan loss (relativistic gan)
            fake_g_pred = self.net_d(self.output)
            l_g_gan = self.cri_gan(fake_g_pred, True, is_disc=False)
            l_g_total += l_g_gan
            loss_dict['l_g_gan'] = l_g_gan

            l_g_total.backward()
            self.optimizer_g.step()

        # optimize net_d
        for p in self.net_d.parameters():
            p.requires_grad = True

        self.optimizer_d.zero_grad()

        # real
        real_d_pred = self.net_d(self.gt)
        l_d_real = self.cri_gan(real_d_pred, True, is_disc=True)
        loss_dict['l_d_real'] = l_d_real
        loss_dict['out_d_real'] = torch.mean(real_d_pred.detach())
        l_d_real.backward()
        # fake
        fake_d_pred = self.net_d(self.output.detach().clone())  # clone for pt1.9
        l_d_fake = self.cri_gan(fake_d_pred, False, is_disc=True)
        loss_dict['l_d_fake'] = l_d_fake
        loss_dict['out_d_fake'] = torch.mean(fake_d_pred.detach())
        l_d_fake.backward()
        self.optimizer_d.step()

        for k in loss_dict.keys():
            loss_dict[k] = loss_dict[k].mean().item()

        return loss_dict

    def validation(self, current_iter, tb_logger):
        self.net_g.eval()
        with torch.no_grad():
            output = self.net_g(self.lq).clamp(0, 1)

        for i in range(4):
            tensor = torch.stack(
                (self.lq[i].detach().cpu(), output[i].detach().cpu(), self.gt[i].detach().cpu()), dim=0)
            img_grid = np.clip(make_grid(tensor, nrow=1, normalize=False).numpy(),
                               0, 1)
            tb_logger(img_grid, f"val_img_train/{i}", current_iter)

    def validation_real(self, val_loader, current_iter, tb_logger):
        size = 0
        self.net_g.eval()
        for batch, (lr, sr) in enumerate(val_loader):
            size += batch

            with torch.no_grad():
                output = self.net_g(sr).clamp(0, 1)

            tensor = torch.stack(
                (lr[0].detach().cpu(), sr[0].detach().cpu(), output[0].detach().cpu()), dim=0)
            img_grid = np.clip(make_grid(tensor, nrow=1, normalize=False).numpy(),
                               0, 1)
            tb_logger(img_grid, f"val_img_real/{batch}", current_iter)
