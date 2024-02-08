from .replay_buffer import ReplayBuffer
import os.path
import torch
import torch.optim as optim
import torch.nn as nn
from HMARL.network import build_network
from collections import OrderedDict
import logging
from HMARL.utils.logger import get_root_logger


class BaseAgent:
    def __init__(self, opt):

        self.opt = opt
        # config
        self.is_train = opt["is_train"]
        self.device = opt["device"]

        # data_config
        self.patch_size = opt["datasets"]["train"]["patch_size"]

        # env_config
        self.sigma_s_low = opt["env"]["sigma_s_low"]
        self.sigma_s_range = opt["env"]["sigma_s_range"]
        self.sigma_r_low = opt["env"]["sigma_r_low"]
        self.sigma_r_range = opt["env"]["sigma_r_range"]
        self.bias = opt["env"]["bias"]
        self.gamma = opt["env"]["gamma"]

        # network_config
        self.ppo = build_network(opt["network"])
        self.ppo = self.model_to_device(self.ppo)

        # agent_config
        self.value_coef = opt["agent"]["value_coef"]
        self.entropy_coef = opt["agent"]["entropy_coef"]
        self.clip_param = opt["agent"]["clip_param"]
        self.max_grad_norm = opt["agent"]["max_grad_norm"]

        # replay_buffer
        self.buffer_capacity =opt["agent"]["replay_buffer"]["buffer_capacity"]

        # trian
        self.ppo_epoch = opt["agent"]["train"]["ppo_epoch"]
        self.batch_size = opt["agent"]["train"]["batch_size"]

        # optim
        lr = opt["agent"]["optim"]["lr"]
        betas = opt["agent"]["optim"]["betas"]
        self.optimizer = optim.Adam(self.ppo.parameters(), lr=lr, betas=betas)

        # scheduler
        lr_gamma = opt["agent"]["scheduler"]["lr_gamma"]
        milestones = opt["agent"]["scheduler"]["milestones"]
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=milestones, gamma=lr_gamma)

        self.replay_buffer = ReplayBuffer(**opt["agent"]["replay_buffer"])

        # log
        self.logger = get_root_logger(logger_name='basicsr', log_level=logging.INFO)
        self.logger.info("智能体初始化完成")

    def save(self, epoch, current_iter, best_psnr):
        if current_iter == -1:
            current_iter = 'latest'
        elif current_iter == -2:
            current_iter = 'best_model'
        state = {'epoch': epoch, 'iter': current_iter, "best_psnr": best_psnr, 'state_dict': self.ppo.state_dict(),
                 'optimizer': self.optimizer.state_dict(), 'scheduler': self.scheduler.state_dict()}
        save_filename = f'{current_iter}.pth'
        save_path = os.path.join(self.opt['path']['checkpoint'], save_filename)
        # 保存状态
        torch.save(state, save_path)

    def load_checkpoint(self, checkpoint):
        try:
            self.ppo.load_state_dict(checkpoint)
        except RuntimeError:
            self.logger.info("参数加载失败,去掉module重新加载...")
            new_state_dict = OrderedDict()
            for k, v in checkpoint.items():
                name = k[7:]  # module字段在最前面，从第7个字符开始就可以去掉module
                new_state_dict[name] = v  # 新字典的key值对应的value一一对应
            self.ppo.load_state_dict(new_state_dict)
        logging.info("成功加载网络参数")

    def resume_training(self, resume_state):
        self.optimizer.load_state_dict(resume_state['optimizer'])
        self.scheduler.load_state_dict(resume_state['scheduler'])
        self.load_checkpoint(resume_state["state_dict"])

    def model_to_device(self, net):
        net = net.to(self.device)
        if self.device == "cuda":
            net = nn.DataParallel(net, device_ids=[0, 1])
        return net

    def isnan_param(self):
        params = [p for p in self.ppo.module.parameters()]
        for param in params:
            if torch.isnan(param).sum() == 0:
                return True
        return False

    def isnan_grad(self):
        params = [p for p in self.ppo.module.parameters() if p.grad is not None]
        for param in params:
            if torch.isnan(param.grad).sum() == 0:
                return True
        return False

    def get_max_grad(self, parameters):
        norms = [p.grad.detach().abs().max().to(self.device) for p in parameters]
        total_norm = norms[0] if len(norms) == 1 else torch.max(torch.stack(norms))
        return total_norm

    def get_max_param(self, parameters):
        norms = [p.detach().abs().max().to(self.device) for p in parameters]
        total_norm = norms[0] if len(norms) == 1 else torch.max(torch.stack(norms))
        return total_norm

