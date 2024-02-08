import numpy as np
from abc import abstractmethod
from HMARL.env.common import get_gradient_map
import matplotlib.pyplot as plt

def plot(img):
    plt.imshow(img)
    plt.show()

class RewardFunction:
    @abstractmethod
    def __call__(self, opt, lr, sr, hr, action) -> dict:
        pass


# l1
class RewardL1(RewardFunction):
    def __call__(self, opt, lr, sr, hr, action) -> dict:
        # Decomposing the result
        sr_gamma = sr[:, 1][:, None, :, :]

        main_reward = -np.abs(sr_gamma - sr_gamma)
        reward = {"reward": main_reward}
        return reward


# l1+sharpen
class RewardL1_sharpen(RewardFunction):
    def __call__(self, opt, lr, sr, hr, action) -> dict:
        #  Decomposing the result
        sr_filter = sr[:, 0][:, None, :, :]
        sr_gamma = sr[:, 1][:, None, :, :]
        # l1 reward
        l1_reward = -np.abs(sr_gamma - hr)
        # sobel
        filter_hf_sobel_response = get_gradient_map(sr_filter / 255., sobel=True, laplace=False)
        hr_hf_sobel_response = get_gradient_map(hr / 255., sobel=True, laplace=False)
        sobel_reward = -np.abs(filter_hf_sobel_response - hr_hf_sobel_response)
        # windows
        win = action[:, 0][:, None, :, :]
        # Reward combination
        reward_end = l1_reward + 20 * sobel_reward + 1 * win
        reward = {"reward": reward_end, "l1_reward": l1_reward, "hf_sobel_response": filter_hf_sobel_response, "sobel_reward": sobel_reward, "win_reward": win}
        return reward



