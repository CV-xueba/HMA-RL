from torch.utils.tensorboard import SummaryWriter
import logging

initialized_logger = {}


class Init_tb_logger:
    def __init__(self, opt):
        self.exp_name = opt['name']
        self.use_tb_logger = opt['logger']['use_tb_logger']
        tb_logger = None
        if opt['logger'].get('use_tb_logger') and 'debug' not in opt['name']:
            tb_logger = SummaryWriter(log_dir=opt["path"]["tb_logger"])
        self.tb_logger = tb_logger

    def __call__(self, log_vars, log_type="", current_iter=-1,):
        if self.tb_logger:
            # tensorboard logger
            if "val_img" in log_type:
                self.tb_logger.add_image(log_type, log_vars, current_iter, dataformats='HWC')
            else:
                for k, v in log_vars.items():
                    v = v.mean() if "reward" in log_type else v
                    self.tb_logger.add_scalar(f'{log_type}/{k}', v, current_iter)

def get_root_logger(logger_name='basicsr', log_level=logging.INFO, log_file=None):
    """Get the root logger.

    The logger will be initialized if it has not been initialized. By default a
    StreamHandler will be added. If `log_file` is specified, a FileHandler will
    also be added.

    Args:
        logger_name (str): root logger name. Default: 'basicsr'.
        log_file (str | None): The log filename. If specified, a FileHandler
            will be added to the root logger.
        log_level (int): The root logger level. Note that only the process of
            rank 0 is affected, while other processes will set the level to
            "Error" and be silent most of the time.

    Returns:
        logging.Logger: The root logger.
    """
    logger = logging.getLogger(logger_name)
    # if the logger has been initialized, just return it
    if logger_name in initialized_logger:
        return logger

    format_str = '%(asctime)s %(levelname)s: %(message)s'
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(logging.Formatter(format_str))
    logger.addHandler(stream_handler)
    logger.propagate = False
    if log_file is not None:
        logger.setLevel(log_level)
        # add file handler
        file_handler = logging.FileHandler(log_file, 'w')
        file_handler.setFormatter(logging.Formatter(format_str))
        file_handler.setLevel(log_level)
        logger.addHandler(file_handler)
    initialized_logger[logger_name] = True
    return logger
