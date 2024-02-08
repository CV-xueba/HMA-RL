import sys
sys.path.append("../")

import torch
from os import path as osp
from SRModel.utils.options import parse_options, copy_opt_file
from SRModel.models import build_model
from SRModel.utils.misc import scandir, make_exp_dirs, get_time_str
from SRModel.utils.logger import Init_tb_logger, get_root_logger
import logging
from SRModel.data.div2k import DIV2K
from SRModel.data.realsr import RealSR
from torch.utils.data import DataLoader


def load_resume_state(opt):
    generate_resume_state_path = None
    disc_resume_state_path = None
    if opt['auto_resume']:
        state_path = opt["path"]["checkpoint"]
        if osp.isdir(state_path):
            states = list(scandir(state_path, suffix='pth', recursive=False, full_path=False))
            if len(states) != 0:
                states = [float(v.split('.pth')[0]) for v in states if v.split('.pth')[0].isdigit()]
                generate_resume_state_path = osp.join(state_path, f'{max(states):.0f}.pth')
                disc_resume_state_path = osp.join(state_path, f'{max(states):.0f}_D.pth')
    else:
        if opt['path'].get('resume_state'):
            generate_resume_state_path = opt['path']['generate_resume_state']
            disc_resume_state_path = opt['path']['disc_resume_state']

    if generate_resume_state_path is None:
        generate_resume_state = None
        disc_resume_state = None
    else:
        generate_resume_state = torch.load(generate_resume_state_path, map_location=torch.device(opt["device"]))
        disc_resume_state = torch.load(disc_resume_state_path, map_location=torch.device(opt["device"]))
    return generate_resume_state, disc_resume_state


def main():
    # Load configuration and set random seed
    args, opt = parse_options()

    #  Load model state
    resume_state = load_resume_state(opt)

    # Create experiment directories and log files
    if resume_state[0] is None:
        make_exp_dirs(opt)

    # Copy important files to the experiment directory
    if opt["is_train"]:
        copy_opt_file(args.opt, opt['path']['experiments_root'])

    #  Initialize log
    log_file = osp.join(opt['path']['log'], f"train_{opt['name']}_{get_time_str()}.log")
    logger = get_root_logger(logger_name='basicsr', log_level=logging.INFO, log_file=log_file)
    tb_logger = Init_tb_logger(opt)

    logger.info(opt['name'])

    # Create generator
    model = build_model(opt)

    if resume_state[0]:  # resume training
        model.resume_training(resume_state)  # handle optimizers and schedulers
        start_epoch = resume_state[0]['epoch']
        current_iter = resume_state[0]['iter']
        best_lpips = resume_state[0]['best_lpips']
    else:
        start_epoch = 0
        current_iter = 0
        best_lpips = 0

    # Create dataloader
    dataset = DIV2K(opt)
    dataloader = DataLoader(dataset=dataset,
                            batch_size=opt["train"]["batch_size"],
                            shuffle=True,
                            num_workers=opt["datasets"]["train"]["num_worker"],
                            drop_last=True
                            )

    # Create validation dataloader
    val_dataset = RealSR(opt)
    val_dataloader = DataLoader(dataset=val_dataset,
                                batch_size=32,
                                shuffle=True,
                                num_workers=opt["datasets"]["val"]["num_worker"]
                                )

    # Set training parameters
    epoch = opt["train"]["epoch"]
    # 20 iterations of training images constitute one epoch
    var = 20

    #  Training
    logger.info(" Start training")
    for i_ep in range(start_epoch, epoch):
        for _ in range(var):
            for batch, (sr, hr) in enumerate(dataloader):
                current_iter += 1
                # update learning rate
                model.update_learning_rate(current_iter, warmup_iter=opt['train'].get('warmup_iter', -1))
                # training
                model.feed_data(sr, hr)
                loss_dict = model.optimize_parameters(current_iter)

                # log
                if current_iter % opt['logger']['print_freq'] == 0:
                    # logger.info(f'[epoch:{i_ep}, iter:{current_iter}][percep_loss:{loss_dict["l_g_percep"]:.3f}]')
                    logger.info(f'[epoch:{i_ep}, iter:{current_iter}]')
                    tb_logger(loss_dict, log_type="loss", current_iter=current_iter)

                # save model
                if current_iter % opt["logger"]["save_checkpoint_freq"] == 0:
                    model.save(i_ep, current_iter, best_lpips)
                    # logger.info(f'[train_iter:{current_iter}][percep_loss:{loss_dict["l_g_percep"]:.3f}]')

                # validation
                if current_iter % opt["logger"]["val_freq"] == 0:
                    # val_train_data
                    model.validation(current_iter, tb_logger)

                    # val_real_data
                    model.validation_real(val_dataloader, current_iter, tb_logger)

        # end of iter

    # end of epoch
    model.save(-1, -1, best_lpips)  # -1
    logger.info('Training completed')


if __name__ == '__main__':
    main()
