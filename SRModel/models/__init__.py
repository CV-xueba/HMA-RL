import importlib


def build_model(opt):
    model_name = opt["model_type"]
    model_module = importlib.import_module('.', package='SRModel.models.{}'.format(model_name))
    Model = getattr(model_module, "SRGANModel")
    model = Model(opt)
    return model
