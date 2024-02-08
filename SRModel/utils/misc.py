import os
import os.path as osp
import time


def get_time_str():
    return time.strftime('%Y%m%d_%H%M%S', time.localtime())


def mkdir(path):
    """mkdirs. If path exists,  create a new one.

    Args:
        path (str): Folder path.
    """
    if not osp.exists(path):
        os.makedirs(path, exist_ok=True)


def make_exp_dirs(opt):
    """Make dirs for experiments."""
    if opt['is_train']:
        mkdir(opt["path"]["scripts"])
        mkdir(opt["path"]["tb_logger"])
        mkdir(opt["path"]["checkpoint"])
        mkdir(opt["path"]["validation"])
        mkdir(opt["path"]["log"])


def make_test_results_dirs(opt):
    """Make dirs for experiments."""
    mkdir(opt["path"]["test_results"])


def scandir(dir_path, suffix=None, recursive=False, full_path=False):
    """Scan a directory to find the interested files.

    Args:
        dir_path (str): Path of the directory.
        suffix (str | tuple(str), optional): File suffix that we are
            interested in. Default: None.
        recursive (bool, optional): If set to True, recursively scan the
            directory. Default: False.
        full_path (bool, optional): If set to True, include the dir_path.
            Default: False.

    Returns:
        A generator for all the interested files with relative paths.
    """

    if (suffix is not None) and not isinstance(suffix, (str, tuple)):
        raise TypeError('"suffix" must be a string or tuple of strings')

    root = dir_path

    def _scandir(dir_path, suffix, recursive):
        for entry in os.scandir(dir_path):
            if not entry.name.startswith('.') and entry.is_file():
                if full_path:
                    return_path = entry.path
                else:
                    return_path = osp.relpath(entry.path, root)

                if suffix is None:
                    yield return_path
                elif return_path.endswith(suffix):
                    yield return_path
            else:
                if recursive:
                    yield from _scandir(entry.path, suffix=suffix, recursive=recursive)
                else:
                    continue

    return _scandir(dir_path, suffix=suffix, recursive=recursive)