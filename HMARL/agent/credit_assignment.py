import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
import torch.optim as optim
from collections import OrderedDict
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler, SequentialSampler
import os
import logging
from HMARL.utils.logger import get_root_logger


def hard_update(target, source):
    for m1, m2 in zip(target.modules(), source.modules()):
        m1._buffers = m2._buffers.copy()
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)


def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(
            target_param.data * (1.0 - tau) + param.data * tau
        )


class UNet(nn.Module):
    def __init__(self, num_in_ch=2, num_feat=64, skip_connection=True):
        super(UNet, self).__init__()
        self.skip_connection = skip_connection
        norm = spectral_norm
        # the first convolution
        self.conv0 = nn.Conv2d(num_in_ch, num_feat, kernel_size=3, stride=1, padding=1)
        # downsample
        self.conv1 = norm(nn.Conv2d(num_feat, num_feat * 2, 4, 2, 1, bias=False))
        self.conv2 = norm(nn.Conv2d(num_feat * 2, num_feat * 4, 4, 2, 1, bias=False))
        self.conv3 = norm(nn.Conv2d(num_feat * 4, num_feat * 8, 4, 2, 1, bias=False))
        # upsample
        self.conv4 = norm(nn.Conv2d(num_feat * 8, num_feat * 4, 3, 1, 1, bias=False))
        self.conv5 = norm(nn.Conv2d(num_feat * 4, num_feat * 2, 3, 1, 1, bias=False))
        self.conv6 = norm(nn.Conv2d(num_feat * 2, num_feat, 3, 1, 1, bias=False))
        # extra convolutions
        self.conv7 = norm(nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=False))
        self.conv8 = norm(nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=False))
        self.conv9 = nn.Conv2d(num_feat, 1, 3, 1, 1)

    def forward(self, x):
        # downsample
        x0 = F.leaky_relu(self.conv0(x), negative_slope=0.2, inplace=True)
        x1 = F.leaky_relu(self.conv1(x0), negative_slope=0.2, inplace=True)
        x2 = F.leaky_relu(self.conv2(x1), negative_slope=0.2, inplace=True)
        x3 = F.leaky_relu(self.conv3(x2), negative_slope=0.2, inplace=True)

        # upsample
        x3 = F.interpolate(x3, scale_factor=2, mode='bilinear', align_corners=False)
        x4 = F.leaky_relu(self.conv4(x3), negative_slope=0.2, inplace=True)

        if self.skip_connection:
            x4 = x4 + x2
        x4 = F.interpolate(x4, scale_factor=2, mode='bilinear', align_corners=False)
        x5 = F.leaky_relu(self.conv5(x4), negative_slope=0.2, inplace=True)

        if self.skip_connection:
            x5 = x5 + x1
        x5 = F.interpolate(x5, scale_factor=2, mode='bilinear', align_corners=False)
        x6 = F.leaky_relu(self.conv6(x5), negative_slope=0.2, inplace=True)

        if self.skip_connection:
            x6 = x6 + x0

        # extra convolutions
        out = F.leaky_relu(self.conv7(x6), negative_slope=0.2, inplace=True)
        out = F.leaky_relu(self.conv8(out), negative_slope=0.2, inplace=True)
        out = self.conv9(out)

        return out


class GANLoss(nn.Module):
    """Define GAN loss.

    Args:
        gan_type (str): Support 'vanilla', 'lsgan', 'wgan', 'hinge'.
        real_label_val (float): The value for real label. Default: 1.0.
        fake_label_val (float): The value for fake label. Default: 0.0.
        loss_weight (float): Loss weight. Default: 1.0.
            Note that loss_weight is only for generators; and it is always 1.0
            for discriminators.
    """

    def __init__(self, gan_type, real_label_val=1.0, fake_label_val=0.0, loss_weight=1.0):
        super(GANLoss, self).__init__()
        self.gan_type = gan_type
        self.loss_weight = loss_weight
        self.real_label_val = real_label_val
        self.fake_label_val = fake_label_val

        if self.gan_type == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif self.gan_type == 'lsgan':
            self.loss = nn.MSELoss()
        elif self.gan_type == 'wgan':
            self.loss = self._wgan_loss
        elif self.gan_type == 'wgan_softplus':
            self.loss = self._wgan_softplus_loss
        elif self.gan_type == 'hinge':
            self.loss = nn.ReLU()
        else:
            raise NotImplementedError(f'GAN type {self.gan_type} is not implemented.')

    def _wgan_loss(self, input, target):
        """wgan loss.

        Args:
            input (Tensor): Input tensor.
            target (bool): Target label.

        Returns:
            Tensor: wgan loss.
        """
        return -input.mean() if target else input.mean()

    def _wgan_softplus_loss(self, input, target):
        """wgan loss with soft plus. softplus is a smooth approximation to the
        ReLU function.

        In StyleGAN2, it is called:
            Logistic loss for discriminator;
            Non-saturating loss for generator.

        Args:
            input (Tensor): Input tensor.
            target (bool): Target label.

        Returns:
            Tensor: wgan loss.
        """
        return F.softplus(-input).mean() if target else F.softplus(input).mean()

    def get_target_label(self, input, target_is_real):
        """Get target label.

        Args:
            input (Tensor): Input tensor.
            target_is_real (bool): Whether the target is real or fake.

        Returns:
            (bool | Tensor): Target tensor. Return bool for wgan, otherwise,
                return Tensor.
        """

        if self.gan_type in ['wgan', 'wgan_softplus']:
            return target_is_real
        target_val = (self.real_label_val if target_is_real else self.fake_label_val)
        return input.new_ones(input.size()) * target_val

    def forward(self, input, target_is_real, is_disc=False):
        """
        Args:
            input (Tensor): The input for the loss module, i.e., the network
                prediction.
            target_is_real (bool): Whether the targe is real or fake.
            is_disc (bool): Whether the loss for discriminators or not.
                Default: False.

        Returns:
            Tensor: GAN loss value.
        """
        target_label = self.get_target_label(input, target_is_real)
        if self.gan_type == 'hinge':
            if is_disc:  # for discriminators in hinge-gan
                input = -input if target_is_real else input
                loss = self.loss(1 + input).mean()
            else:  # for generators in hinge-gan
                loss = -input.mean()
        else:  # other gan types
            loss = self.loss(input, target_label)

        # loss_weight is always 1.0 for discriminators
        return loss if is_disc else loss * self.loss_weight


class CreditAssignment:
    def __init__(self, opt):
        self.opt = opt

        self.CCAnet = UNet(1)
        self.gan_loss = GANLoss(gan_type="vanilla")

        self.device = opt["device"]
        self.gail_epoch = opt["disc"]["train"]["gail_epoch"]
        self.batch_size = opt["disc"]["train"]["batch_size"]
        self.gan_reward_weight = opt["disc"]["train"]["gan_reward_weight"]

        lr = opt["disc"]["optim"]["lr"]
        betas = opt["disc"]["optim"]["betas"]
        self.CCAnet = self.model_to_device(self.CCAnet)
        self.optimizer = optim.Adam(self.CCAnet.parameters(), lr=lr, betas=betas)

        self.logger = get_root_logger(logger_name='basicsr', log_level=logging.INFO)

    def save(self, current_iter):
        if current_iter == -1:
            current_iter = 'latest'
        elif current_iter == -2:
            current_iter = 'best_model'
        state = {'state_dict': self.CCAnet.state_dict(), 'optimizer': self.optimizer.state_dict()}
        save_filename = f'{current_iter}_cca.pth'
        save_path = os.path.join(self.opt['path']['checkpoint'], save_filename)
        torch.save(state, save_path)

    def load_checkpoint(self, checkpoint):
        try:
            self.CCAnet.load_state_dict(checkpoint)
        except RuntimeError:
            new_state_dict = OrderedDict()
            for k, v in checkpoint.items():
                name = k[7:]
                new_state_dict[name] = v
            self.CCAnet.load_state_dict(new_state_dict)

    def resume_training(self, resume_state):
        self.optimizer.load_state_dict(resume_state['optimizer'])
        self.load_checkpoint(resume_state["state_dict"])

    def model_to_device(self, net):
        net = net.to(self.device)
        if self.device == "cuda":
            net = nn.DataParallel(net, device_ids=[0, 1])
        return net

    def update(self, repaly_buffer):
        gan_dict = OrderedDict()
        gan_dict["fake"] = 0
        gan_dict["real"] = 0
        train_step = 0

        for i in range(self.gail_epoch):
            for index in BatchSampler(
                    SubsetRandomSampler(range(len(repaly_buffer))), self.batch_size, False):
                state, next_state, old_action, old_action_log_prob, reward, old_value, adv = repaly_buffer.sample(
                    index)

                self.optimizer.zero_grad()
                fake = next_state[:, 0].unsqueeze(1)
                real = next_state[:, 1].unsqueeze(1)

                real = self.CCAnet(real)
                fake = self.CCAnet(fake)

                self.optimizer.zero_grad()
                cost = (self.gan_loss(fake, False) + self.gan_loss(real, True)) / 2
                cost.backward()
                self.optimizer.step()

                self.optimizer.step()

                train_step += 1

                # log
                gan_dict["fake"] += fake.mean()
                gan_dict["real"] += real.mean()

        for k in gan_dict.keys():
            gan_dict[k] /= train_step

        return gan_dict

    def predict_rewards(self, repaly_buffer):
        new_reward_dict = OrderedDict()
        new_reward_dict["cca"] = 0
        new_reward_dict["l1"] = 0
        new_reward_dict["new_reward"] = 0
        per_step = 0

        next_state = repaly_buffer.next_state_buffer.to(self.device)
        pre_reward = repaly_buffer.reward_buffer
        for index in BatchSampler(
                SequentialSampler(range(len(repaly_buffer))), self.batch_size, False):
            with torch.no_grad():
                fake_data = next_state[index, 0].unsqueeze(1)
                l1_reward = pre_reward[index]
                cca_reward = self.CCAnet(fake_data).cpu()
                new_reward = l1_reward + self.gan_reward_weight * cca_reward
                pre_reward[index] = new_reward

            new_reward_dict["l1"] += l1_reward.mean()
            new_reward_dict["cca"] += cca_reward.mean()
            per_step += 1

        for k in new_reward_dict.keys():
            new_reward_dict[k] /= per_step
        new_reward_dict["new_reward"] = repaly_buffer.reward_buffer.mean()
        return new_reward_dict
