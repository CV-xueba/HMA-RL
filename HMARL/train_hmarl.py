import sys
sys.path.append("../")

import torch
from env.env_model import build_env
from os import path as osp
from utils.options import parse_options, copy_opt_file
from agent import build_agent, build_cca
from utils.misc import scandir, make_exp_dirs, get_time_str
from utils.logger import Init_tb_logger, get_root_logger
import logging
from env.data.srdata_base import SRData
from torch.utils.data import DataLoader


def load_resume_state(opt):
    agent_resume_state_path = None
    disc_resume_state_path = None
    if opt['auto_resume']:
        state_path = opt["path"]["checkpoint"]
        if osp.isdir(state_path):
            states = list(scandir(state_path, suffix='pth', recursive=False, full_path=False))
            if len(states) != 0:
                states = [float(v.split('.pth')[0]) for v in states if v.split('.pth')[0].isdigit()]
                agent_resume_state_path = osp.join(state_path, f'{max(states):.0f}.pth')
                disc_resume_state_path = osp.join(state_path, f'{max(states):.0f}_D.pth')
    else:
        if opt['path'].get('resume_state'):
            agent_resume_state_path = opt['path']['agent_resume_state']
            disc_resume_state_path = opt['path']['disc_resume_state']

    if agent_resume_state_path is None:
        agent_resume_state = None
        disc_resume_state = None
    else:
        agent_resume_state = torch.load(agent_resume_state_path, map_location=torch.device(opt["device"]))
        disc_resume_state = torch.load(disc_resume_state_path, map_location=torch.device(opt["device"]))
    return agent_resume_state, disc_resume_state


def main():
    # Load configuration and set random seed
    args, opt = parse_options()

    # Load model state
    resume_state = load_resume_state(opt)

    # Create experiment directory and log files
    # if resume_state[0] is None:
    make_exp_dirs(opt)

    # Copy important files to experiment directory
    if opt["is_train"]:
        copy_opt_file(args.opt, opt['path']['experiments_root'])

    # Initialize logs
    log_file = osp.join(opt['path']['log'], f"train_{opt['name']}_{get_time_str()}.log")
    logger = get_root_logger(logger_name='basicsr', log_level=logging.INFO, log_file=log_file)
    tb_logger = Init_tb_logger(opt)

    # Create environment
    env = build_env(opt)

    # Create agent
    agent = build_agent(opt)

    # Create convolutional credit assignment network
    cca = build_cca(opt)

    if resume_state[0]:  # resume training
        agent.resume_training(resume_state[0])  # handle optimizers and schedulers
        cca.resume_training(resume_state[1])  # handle optimizers
        start_epoch = resume_state[0]['epoch']
        current_iter = resume_state[0]['iter']
        best_psnr = resume_state[0]['best_psnr']
    else:
        start_epoch = 0
        current_iter = 0
        best_psnr = 0

    # creat dataloader
    dataset = SRData(opt)
    dataloader = DataLoader(dataset=dataset,
                            batch_size=opt["train"]["batch_size"],
                            shuffle=True,
                            num_workers=opt["datasets"]["train"]["num_worker"],
                            drop_last=True
                            )

    # Set training parameters
    epoch = opt["train"]["epoch"]
    # Iterate through training images 20 times for one epoch
    var = 20

    # Training
    logger.info("Start training")
    for i_ep in range(start_epoch, epoch):
        for _ in range(var):
            for batch, (lr, hr) in enumerate(dataloader):
                current_iter += 1
                #  Environment interaction
                agent.ppo.eval()
                state = env.reset(lr, hr)
                action, action_log_prob, value, action_sample, gamma_noise = agent.select_action(state)
                next_state, reward = env.step(action, gamma_noise)
                agent.replay_buffer.push(state, next_state, action_sample, action_log_prob, reward['reward'], value)

                # log reward
                tb_logger(reward, log_type="reward", current_iter=current_iter)
                if current_iter % opt["logger"]["print_freq"] == 0:
                    logger.info(f'[epoch:{i_ep}, iter:{current_iter}][reward:{reward["reward"].mean():.3f}]')

                # Update model
                if agent.replay_buffer.isFull():
                    # disc update
                    logger.info(f'[epoch:{i_ep}, iter:{current_iter}][更新判别器]]')
                    gan_dict = cca.update(agent.replay_buffer)
                    new_reward_dict = cca.predict_rewards(agent.replay_buffer)
                    logger.info(f'[epoch:{i_ep}, iter:{current_iter}][grad:{new_reward_dict["gan"]}]')
                    # agent update
                    agent.ppo.train()
                    logger.info(f'[epoch:{i_ep}, iter:{current_iter}][更新智能体]]')
                    loss_dict, value_dict, lr_dict = agent.update()

                    # log loss
                    tb_logger(new_reward_dict, log_type="new_reward", current_iter=current_iter)
                    tb_logger(gan_dict, log_type="gan", current_iter=current_iter)
                    tb_logger(loss_dict, log_type="loss", current_iter=current_iter)
                    tb_logger(value_dict, log_type="value", current_iter=current_iter)
                    tb_logger(lr_dict, log_type="lr", current_iter=current_iter)

                    if current_iter % opt["logger"]["save_checkpoint_freq"] == 0:
                        logger.info(f'[train_iter:{current_iter}]')

                # Save model periodically
                if current_iter % opt['logger']['save_checkpoint_freq'] == 0:
                    agent.save(i_ep, current_iter, best_psnr)
                    cca.save(current_iter)

        # end of iter

    # end of epoch
    agent.save(-1, -1, best_psnr)  # -1
    logger.info('Training finished')


if __name__ == '__main__':
    main()
