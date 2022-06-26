from dis import dis
from distutils.command import config
from gym import Env
import math
import numpy as np
from numpy import roll
from tomlkit import value
import torch
from src.ppo.model import DQN
from collections import namedtuple, deque
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import time
from IPython.display import clear_output
import wandb
from torch.distributions import MultivariateNormal
from torch.distributions.categorical import Categorical

Config = namedtuple(
    'Config', ('training_step', 'lr', 'lr_critic', 'lr_decay', 'gamma', 'loss', 'n_updates', 'batch_size',  'epsilon', 'clip'))

Rollout = namedtuple(
    'Rollout', ('states', 'actions', 'probs', 'rewards', 'rewards_togo', 'ep_lens'))


def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)


class PPOAgent():
    def __init__(self, env, config: Config = {}, id: str = ""):
        self.env = env
        self.num_actions = self.env.action_space.n

        self.actor_net = DQN(self.env.observation_space.n,
                             self.num_actions, softmax=True)
        self.critic_net = DQN(self.env.observation_space.n, 1)
        # self.actor_net.apply(init_weights)
        # self.critic_net.apply(init_weights)
        #self.actor_net = DQN(4, self.num_actions, softmax=True)
        #self.critic_net = DQN(4, 1)
        self.t_step = 0
        if id != "":
            self.id = str(id) + "-" + str(int(time.time()))
        else:
            self.id = str(int(time.time()))

        if config:
            self.config = config
            self.batch_size = config.batch_size
            self.training_step = config.training_step
            self.gamma = config.gamma
            self.n_updates = config.n_updates
            self.loss_type = config.loss
            self.clip = config.clip
            self.actor_optimizer = torch.optim.Adam(
                params=self.actor_net.parameters(), lr=config.lr)  # TODO try other optimizer
            self.critic_optimizer = torch.optim.Adam(
                params=self.critic_net.parameters(), lr=config.lr_critic)  # TODO try other optimizer
            # Create the covariance matrix using for env exploration
            self.cov_mat = torch.diag(torch.full(
                size=(self.num_actions,), fill_value=config.epsilon))

        self.device = torch.device("cpu")

    def fit(self, wandb=False):
        t = 0
        scores = []
        while t < self.training_step:
            rollout = self._get_rollout_data()
            scores = [*scores, *rollout.rewards]
            qvalues, _ = self._evaluate_states(rollout.states)
            # Calculate advantage as (actor Qvalues - critic Qvalues) and normalize it
            advantage = rollout.rewards_togo - qvalues.detach()
            advantage = (advantage - advantage.mean()) / \
                (advantage.std() + 1e-10)  # TODO check normalize method
            # Update actor & critic network
            for _ in range(self.n_updates):
                qvalues, current_probs = self._evaluate_states(
                    rollout.states, rollout.actions)
                #ratios = torch.exp(current_probs - rollout.probs)
                ratios = current_probs.exp() - rollout.probs.exp()
                surrogate_loss1 = ratios * advantage
                surrogate_loss2 = torch.clamp(
                    ratios, 1 - self.clip, 1 + self.clip) * advantage
                # Calculate actor loss (optimize Adam minizes the loss,
                #  so minimizing the negative loss maximizes the performance function)
                actor_loss = (-torch.min(surrogate_loss1,
                              surrogate_loss2)).mean()
                # Calc gradients and perform backward propagation for actor network
                self.actor_optimizer.zero_grad()
                # Retain graph to prevent buffer to be freed
                actor_loss.backward(retain_graph=True)
                self.actor_optimizer.step()

                # Updating critic parameters with mse of the predicted values at the current epoch with rewards-to-go
                critic_loss = nn.functional.mse_loss(
                    qvalues, rollout.rewards_togo)
                # Calc gradients and perform backward propagation for critic network
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                self.critic_optimizer.step()
            # Add how many step we played
            t += np.sum(rollout.ep_lens)
        return scores

    def _select_action(self, state):
        """
        Select action using actor network
        """
        #state = torch.FloatTensor(state).to(self.device)
        state = torch.tensor([state], device=self.device)
        dist = self.actor_net(state)
        #dist = Categorical(dist)

        action = dist.sample()
        prob = dist.log_prob(action)
        return action.item(), prob.detach()

    def _compute_rewards_togo(self, rewards):
        """
        Reduce the variance of policy gradient. 
        We use causality, and remove part of the sum over rewards so that 
        only actions happened after the reward are taken into account.
        return array of our actor network Q values
        """
        rewards_togo = []
        for episode_rewards in reversed(rewards):
            discounted_reward = 0
            for reward in reversed(episode_rewards):
                discounted_reward = reward + discounted_reward * self.gamma
                # Insert reward at index 0
                rewards_togo.insert(0, discounted_reward)
        return rewards_togo

    def _evaluate_states(self, states, actions=[]):
        """
        Evaluate states using critic network
        return array of our critic network Q values
        """
        values = self.critic_net(states).squeeze()  # TODO print that
        probs = []
        if len(actions) > 0:
            # Create our Multivariate Normal Distribution and get action prob
            dist = self.actor_net(states)  # TODO print mean
           #dist = Categorical(dist)
            probs = dist.log_prob(actions)
            ##dist = MultivariateNormal(mean, self.cov_mat)
            ##prob = dist.log_prob(actions)
        return values, probs

    def _get_rollout_data(self) -> Rollout:
        """
        Get batch data by playing the game for self.batch_size timestep
        """
        max_step = 100
        t = 0
        states, actions, probs, rewards, ep_lens = [], [], [], [], []
        while t < self.batch_size:
            episode_rewards = []
            state = self.env.reset()
            done = False
            step = 0
            while step < max_step:
                # while done == False:
                step += 1
                t += 1
                states.append(state)
                action, prob = self._select_action(state)
                state, reward, done, _ = self.env.step(action)
                # Save data
                episode_rewards.append(reward)
                actions.append(action)
                probs.append(prob)
                self.t_step += 1
                if done:
                    break
            ep_lens.append(step + 1)
            rewards.append(episode_rewards)
            print("\rEpisode reward: {0}".format(
                np.sum(episode_rewards)), end="\r")

        rollout = Rollout(states=torch.tensor(states, device=self.device),
                          actions=torch.tensor(actions, device=self.device),
                          probs=torch.tensor(probs, device=self.device),
                          rewards=rewards,
                          rewards_togo=torch.tensor(
                              self._compute_rewards_togo(rewards), device=self.device),
                          ep_lens=ep_lens)

        return rollout
