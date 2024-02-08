import importlib


class ComputeReward:
    def __init__(self, opt):
        self.opt = opt
        reward_function_name = opt["reward"]["name"]
        reward_function_module = importlib.import_module('.', package='HMARL.env.reward.reward_function')
        self.reward_function = getattr(reward_function_module, reward_function_name)()
        # harr loss

    def __call__(self, sr, hr, lr, action):
        # reward
        reward = self.reward_function(self.opt, lr, sr, hr, action)
        return reward
