import torch


class ReplayBuffer:
    def __init__(self, buffer_capacity, state_num, action_num, patch_size, batch_size, device):
        self.state_buffer = torch.zeros((buffer_capacity, state_num, patch_size, patch_size), dtype=torch.float)
        self.next_state_buffer = torch.zeros((buffer_capacity, 2 * state_num, patch_size, patch_size), dtype=torch.float)
        self.action_buffer = torch.zeros((buffer_capacity, action_num, patch_size, patch_size), dtype=torch.float)
        self.action_log_prob_buffer = torch.zeros((buffer_capacity, action_num, patch_size, patch_size), dtype=torch.float)
        self.reward_buffer = torch.zeros((buffer_capacity, 1, patch_size, patch_size), dtype=torch.float)
        self.value_buffer = torch.zeros((buffer_capacity, 1, patch_size, patch_size), dtype=torch.float)
        self.advantages = torch.zeros((buffer_capacity, 1, patch_size, patch_size), dtype=torch.float)

        self.__top = 0
        self.__capacity = buffer_capacity
        self.__device = device
        self.__batch_size = batch_size

    def push(self, state, next_state, action, action_log_prob, reward, value):
        batch_size = state.shape[0]
        if self.__top + batch_size >= self.__capacity:
            batch_size = self.__capacity - self.__top

        self.state_buffer[self.__top:self.__top + batch_size] = torch.from_numpy(state[0: batch_size])
        self.next_state_buffer[self.__top:self.__top + batch_size] = torch.from_numpy(next_state[0: batch_size])
        self.action_buffer[self.__top:self.__top + batch_size] = torch.from_numpy(action[0: batch_size])
        self.action_log_prob_buffer[self.__top:self.__top + batch_size] = torch.from_numpy(
            action_log_prob[0: batch_size])
        self.reward_buffer[self.__top:self.__top + batch_size] = torch.from_numpy(reward[0: batch_size])
        self.value_buffer[self.__top:self.__top + batch_size] = torch.from_numpy(value[0: batch_size])

        self.__top = (self.__top + batch_size)

    def sample(self, index):
        state_batch = self.state_buffer[index].to(self.__device).float()
        next_state_batch = self.next_state_buffer[index].to(self.__device).float()
        action_batch = self.action_buffer[index].to(self.__device).float()
        action_log_prob_batch = self.action_log_prob_buffer[index].to(self.__device).float()
        reward_batch = self.reward_buffer[index].to(self.__device).float()
        value_batch = self.value_buffer[index].to(self.__device).float()
        adv_batch = self.advantages[index].to(self.__device).float()

        return state_batch, next_state_batch, action_batch, action_log_prob_batch, reward_batch, value_batch, adv_batch

    def reward_mean(self):
        self.reward_buffer = self.reward_buffer / (self.reward_buffer.std() + 1e-8)

    def clear(self):
        self.__top = 0

    def normal_adv(self):
        self.reward_mean()
        self.advantages = (self.reward_buffer - self.value_buffer).detach()
        self.advantages = (self.advantages - self.advantages.mean()) / (self.advantages.std() + 1e-10)

    def isFull(self):
        return self.__top == self.__capacity

    def __len__(self):
        return self.__top
