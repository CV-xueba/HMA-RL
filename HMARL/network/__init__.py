import importlib


def build_network(opt):
    network_name = opt.pop("name")
    network_module = importlib.import_module('.', package='HMARL.network.{}'.format(network_name))
    PPO = getattr(network_module, "PPO")
    net = PPO(**opt)
    return net
