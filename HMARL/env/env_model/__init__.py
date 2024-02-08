import importlib


def build_env(opt):
    env_name = opt["env"]["name"]
    env_module = importlib.import_module('.', package='HMARL.env.env_model.{}'.format(env_name))
    Env = getattr(env_module, "SuperResolutionEnv")
    env = Env(opt)
    return env
