import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from torch.distributions import Beta
from collections import OrderedDict
from .agent_base import BaseAgent


class Agent(BaseAgent):
    def __init__(self, opt):
        super(Agent, self).__init__(opt)

    def action_adapter(self, action):
        action[:, 0, :, :] = action[:, 0, :, :] * self.sigma_s_range + self.sigma_s_low
        action[:, 1, :, :] = action[:, 1, :, :] * self.sigma_r_range + self.sigma_r_low
        action[:, 2, :, :] = (2 * action[:, 2, :, :] - 1) * self.bias
        action[:, 3, :, :] = (2 * action[:, 3, :, :] - 1) * self.gamma
        return action

    def select_action(self, state):
        state = torch.from_numpy(state).float().to(self.device)
        with torch.no_grad():
            policy_alpha, policy_beta, value = self.ppo(state)

        # Action sampling
        action_dist = Beta(policy_alpha, policy_beta)
        action_sample = action_dist.sample()
        action_mean = (policy_alpha / (policy_alpha + policy_beta))

        # Probability
        action_log_prob = action_dist.log_prob(action_sample)

        # Mapping to environment actions
        action_env = self.action_adapter(action_sample.clone())
        action_env_mean = self.action_adapter(action_mean.clone())

        gamma_noise = (action_env - action_env_mean)[:, 3]

        return action_env.cpu().numpy(), action_log_prob.cpu().numpy(), value.cpu().numpy(), action_sample.cpu().numpy(), gamma_noise.cpu().numpy()

    def select_action_play(self, state):
        state = torch.from_numpy(state).float().to(self.device)
        with torch.no_grad():
            policy_alpha, policy_beta, value = self.ppo(state)

        action = (policy_alpha / (policy_alpha + policy_beta)).cpu().numpy()
        action = self.action_adapter(action)
        return action

    def update(self):
        loss_dict = OrderedDict()
        value_dict = OrderedDict()

        lr_dict = OrderedDict()
        loss_dict["action_loss_mean"] = 0
        loss_dict["value_loss_mean"] = 0
        loss_dict["entropy_loss_mean"] = 0
        loss_dict["loss_mean"] = 0
        value_dict["value_mean"] = 0
        value_dict["adv_mean"] = 0
        lr_dict["lr"] = 0
        train_step = 0

        # Normalizing the advantage function
        self.replay_buffer.normal_adv()

        for i in range(self.ppo_epoch):
            for index in BatchSampler(
                    SubsetRandomSampler(range(self.buffer_capacity)), self.batch_size, False):

                state, next_state, old_action, old_action_log_prob, reward, old_value, adv = self.replay_buffer.sample(index)

                policy_alpha, policy_beta, value = self.ppo(state)

                # Probability distribution
                action_dist = Beta(policy_alpha, policy_beta)

                #  Variance of action probability
                action_log_prob = action_dist.log_prob(old_action)
                action_dist_entropy = action_dist.entropy()

                # action loss clip
                ratio = torch.exp(action_log_prob - old_action_log_prob.detach())
                surr_1 = ratio * adv
                surr_2 = torch.clamp(ratio, 1.0 - self.clip_param,
                                     1.0 + self.clip_param) * adv

                self.optimizer.zero_grad()

                # action loss
                action_loss = -torch.mean(torch.min(surr_1, surr_2))

                # value loss clip
                value_clip = old_value + torch.clamp(value - old_value, -self.clip_param, self.clip_param)
                value_loss_1 = F.mse_loss(value, reward)
                value_loss_2 = F.mse_loss(value_clip, reward)
                value_loss = torch.max(value_loss_1, value_loss_2)

                # entropy_loss
                entropy_loss = -torch.mean(action_dist_entropy)

                loss = action_loss + value_loss * self.value_coef + entropy_loss * self.entropy_coef

                if self.device == "cuda":
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.ppo.module.parameters(), self.max_grad_norm)
                    self.optimizer.step()
                    self.scheduler.step()

                train_step += 1

                # log
                loss_dict["action_loss_mean"] += action_loss
                loss_dict["value_loss_mean"] += value_loss
                loss_dict["entropy_loss_mean"] += entropy_loss
                loss_dict["loss_mean"] += loss
                value_dict["value_mean"] += value.mean()
                value_dict["adv_mean"] += adv.mean()


        for k in loss_dict.keys():
            loss_dict[k] /= train_step
        for k in value_dict.keys():
            value_dict[k] /= train_step
        lr_dict["lr"] = self.scheduler.get_last_lr()[-1]

        self.replay_buffer.clear()

        return loss_dict, value_dict, lr_dict

