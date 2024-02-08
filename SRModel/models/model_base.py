import os.path
import torch
import torch.optim as optim
import torch.nn as nn
from SRModel.network import build_network
from SRModel.losses import build_loss
from collections import OrderedDict
import logging
from SRModel.utils.logger import get_root_logger
from SRModel.metrics.measure import IQA


class BASEModel:
    def __init__(self, opt):

        self.opt = opt
        # config
        self.is_train = opt["is_train"]
        self.device = opt["device"]

        # data_config
        self.patch_size = opt["datasets"]["train"]["patch_size"]

        # define network
        self.net_g = build_network(opt['network_g'])
        self.net_g = self.model_to_device(self.net_g)

        self.net_d = build_network(self.opt['network_d'])
        self.net_d = self.model_to_device(self.net_d)

        self.net_g.train()
        self.net_d.train()

        # define loss
        self.cri_perceptual = build_loss(opt["train"]["perceptual_opt"]).to(self.device)
        self.cri_gan = build_loss(opt["train"]["gan_opt"]).to(self.device)

        self.net_d_iters = opt["train"]['net_d_iters']
        self.net_d_init_iters = opt["train"]['net_d_init_iters']

        # define metrics
        self.measure = IQA(metrics=["psnr", "ssim", "lpips"])

        # trian
        self.batch_size = opt["train"]["batch_size"]

        # optim_g
        lr_g = opt["train"]["optim_g"]["lr"]
        betas_g = opt["train"]["optim_g"]["betas"]
        self.optimizer_g = optim.Adam(self.net_g.parameters(), lr=lr_g, betas=betas_g)

        # optim_d
        lr_d = opt["train"]["optim_d"]["lr"]
        betas_d = opt["train"]["optim_d"]["betas"]
        self.optimizer_d = optim.Adam(self.net_d.parameters(), lr=lr_d, betas=betas_d)
        self.optimizers = [self.optimizer_g, self.optimizer_d]

        # scheduler
        gamma = opt["train"]["scheduler"]["gamma"]
        milestones = opt["train"]["scheduler"]["milestones"]

        self.scheduler_g = torch.optim.lr_scheduler.MultiStepLR(self.optimizer_g, milestones=milestones, gamma=gamma)
        self.scheduler_d = torch.optim.lr_scheduler.MultiStepLR(self.optimizer_d, milestones=milestones, gamma=gamma)
        self.schedulers = [self.scheduler_g, self.scheduler_d]

        # train_data
        self.lq = None
        self.gt = None

        # log
        self.logger = get_root_logger(logger_name='basicsr', log_level=logging.INFO)
        self.logger.info("智能体初始化完成")

    def _get_init_lr(self):
        """Get the initial lr, which is set by the scheduler.
        """
        init_lr_groups_l = []
        for optimizer in self.optimizers:
            init_lr_groups_l.append([v['initial_lr'] for v in optimizer.param_groups])
        return init_lr_groups_l

    def _set_lr(self, lr_groups_l):
        """Set learning rate for warmup.

        Args:
            lr_groups_l (list): List for lr_groups, each for an optimizer.
        """
        for optimizer, lr_groups in zip(self.optimizers, lr_groups_l):
            for param_group, lr in zip(optimizer.param_groups, lr_groups):
                param_group['lr'] = lr

    def update_learning_rate(self, current_iter, warmup_iter=-1):
        """Update learning rate.

        Args:
            current_iter (int): Current iteration.
            warmup_iter (int)： Warmup iter numbers. -1 for no warmup.
                Default： -1.
        """
        if current_iter > 1:
            for scheduler in self.schedulers:
                scheduler.step()
        # set up warm-up learning rate
        if current_iter < warmup_iter:
            # get initial lr for each group
            init_lr_g_l = self._get_init_lr()
            # modify warming-up learning rates
            # currently only support linearly warm up
            warm_up_lr_l = []
            for init_lr_g in init_lr_g_l:
                warm_up_lr_l.append([v / warmup_iter * current_iter for v in init_lr_g])
            # set learning rate
            self._set_lr(warm_up_lr_l)

    def save(self, epoch, current_iter, best_lpips):
        if current_iter == -1:
            current_iter = 'latest'
        elif current_iter == -2:
            current_iter = 'best_model'

        # 生成器
        state_g = {'epoch': epoch, 'iter': current_iter, "best_lpips": best_lpips,
                   'state_dict': self.net_g.state_dict(),
                   'optimizer': self.optimizer_g.state_dict(), 'scheduler': self.scheduler_g.state_dict()}
        save_filename_g = f'{current_iter}.pth'
        save_path_g = os.path.join(self.opt['path']['checkpoint'], save_filename_g)
        torch.save(state_g, save_path_g)

        # 判别器
        state_d = {'state_dict': self.net_d.state_dict(), 'optimizer': self.optimizer_d.state_dict(),
                   'scheduler': self.scheduler_d.state_dict()}
        save_filename_d = f'{current_iter}_D.pth'
        save_path_d = os.path.join(self.opt['path']['checkpoint'], save_filename_d)
        # 保存状态

        torch.save(state_d, save_path_d)

    def load_checkpoint(self, checkpoint_g, checkpoint_d):
        try:
            self.net_g.load_state_dict(checkpoint_g)
            self.net_d.load_state_dict(checkpoint_d)
        except RuntimeError as e:
            print(e)
            self.logger.info("参数加载失败,去掉module重新加载...")

            # 加载生成器
            new_state_dict = OrderedDict()
            for k, v in checkpoint_g.items():
                name = k[7:]  # module字段在最前面，从第7个字符开始就可以去掉module
                new_state_dict[name] = v  # 新字典的key值对应的value一一对应
            self.net_g.load_state_dict(new_state_dict)

            # 加载判别器
            new_state_dict = OrderedDict()
            for k, v in checkpoint_d.items():
                name = k[7:]  # module字段在最前面，从第7个字符开始就可以去掉module
                new_state_dict[name] = v  # 新字典的key值对应的value一一对应
            self.net_d.load_state_dict(new_state_dict)

        logging.info("成功加载网络参数")

    def resume_training(self, resume_state):
        self.optimizer_g.load_state_dict(resume_state[0]['optimizer'])
        self.optimizer_d.load_state_dict(resume_state[1]['optimizer'])
        self.scheduler_g.load_state_dict(resume_state[0]['scheduler'])
        self.scheduler_d.load_state_dict(resume_state[1]['scheduler'])
        self.load_checkpoint(resume_state[0]["state_dict"], resume_state[1]["state_dict"])

    def model_to_device(self, net):
        net = net.to(self.device)
        if self.device == "cuda":
            net = nn.DataParallel(net, device_ids=[0, 1])
        return net

    def feed_data(self, lr, hr):
        self.lq = lr.to(self.device)
        self.gt = hr.to(self.device)



