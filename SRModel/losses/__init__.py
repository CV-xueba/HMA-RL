from copy import deepcopy
import importlib


def build_loss(opt):
    opt = deepcopy(opt)
    loss_name = opt.pop("type")
    loss_module = importlib.import_module('.', package='SRModel.losses.losses')
    Loss = getattr(loss_module, loss_name)
    loss = Loss(**opt)
    return loss

