import importlib
import copy
from SRModel.network.edsr_unet_skip_connection import EDSRUNetSkipConnection
from SRModel.network.discriminator_arch_spectral_norm import UNetDiscriminatorSN


def build_network(opt):
    opt = copy.deepcopy(opt)
    network_name = opt.pop("type")
    network_module = importlib.import_module('.', package='SRModel.network')
    Net = getattr(network_module, network_name)
    net = Net(**opt)
    return net
