import yaml
import argparse
import os.path as osp
import numpy as np
import random
import torch
import os


def set_random_seed(seed):
    """Set random seeds."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_options(test=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', type=str, default="configs/train_config.yml", help='Path to option YAML file.')
    parser.add_argument('--auto_resume', action='store_true')
    args = parser.parse_args()

    # parse yml to dict
    if test:
        opt_path = test
        for file in os.listdir(opt_path):
            if osp.splitext(file)[1] == '.yml':
                args.opt = osp.join(opt_path, file)

    with open(args.opt, mode='r') as f:
        opt = yaml.load(f, Loader=yaml.FullLoader)

    opt['auto_resume'] = args.auto_resume

    # random seed
    seed = opt.get('manual_seed')
    if seed is None:
        seed = random.randint(1, 10000)
        print("random_seed:", seed)
        opt['manual_seed'] = seed
    set_random_seed(seed)

    # 环境参数
    action_num = opt["env"]["action_num"]

    # 添加网络参数
    opt["network"]["state_num"] = opt["env"]["state_num"]
    opt["network"]["action_num"] = action_num

    # 添加 replay_buffer 参数
    opt["agent"]["replay_buffer"]["state_num"] = opt["env"]["state_num"]
    opt["agent"]["replay_buffer"]["action_num"] = action_num
    opt["agent"]["replay_buffer"]["patch_size"] = opt["datasets"]["train"]["patch_size"]
    opt["agent"]["replay_buffer"]["batch_size"] = opt["train"]["batch_size"]
    opt["agent"]["replay_buffer"]["device"] = opt["device"]

    opt["path"]["experiments_root"] = test
    # 添加路径
    opt["path"]["scripts"] = osp.join(opt["path"]["experiments_root"], "scripts")
    opt["path"]["tb_logger"] = osp.join(opt["path"]["experiments_root"], "tb_logger")
    opt["path"]["checkpoint"] = osp.join(opt["path"]["experiments_root"], "checkpoint")
    opt['path']['validation'] = osp.join(opt["path"]["experiments_root"], "validation")
    opt['path']['log'] = osp.join(opt["path"]["experiments_root"], "log")

    if test:
        opt['is_train'] = False
    else:
        opt['is_train'] = True

    return args, opt


def copy_opt_file(opt_file, experiments_root):
    # copy the yml file to the experiment root
    import sys
    import time
    from shutil import copyfile
    cmd = ' '.join(sys.argv)
    filename = osp.join(experiments_root, osp.basename(opt_file))
    copyfile(opt_file, filename)

    with open(filename, 'r+') as f:
        lines = f.readlines()
        lines.insert(0, f'# GENERATE TIME: {time.asctime()}\n# CMD:\n# {cmd}\n\n')
        f.seek(0)
        f.writelines(lines)

