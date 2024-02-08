import importlib


def build_agent(opt):
    agent_name = opt["agent"]["name"]
    agent_module = importlib.import_module('.', package='HMARL.agent.{}'.format(agent_name))
    Agent = getattr(agent_module, "Agent")
    agent = Agent(opt)
    return agent


def build_cca(opt):
    cca_name = opt["cca"]["name"]
    cca_module = importlib.import_module('.', package='HMARL.agent.{}'.format(cca_name))
    CCA = getattr(cca_module, "Discriminator")
    cca = CCA(opt)
    return cca
