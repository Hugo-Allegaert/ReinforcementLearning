import math
import random
import time
from collections import namedtuple, deque

import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from IPython.display import clear_output

from src.memory import ReplayMemory, Experience
from src.model import DQN

Config = namedtuple(
    'Config', (
        'target_update', 'lr', 'lr_min', 'lr_decay', 'gamma', 'loss', 'memory_size', 'batch_size', 'eps_start',
        'eps_min', 'eps_decay', 'learning_start', 'double_dqn'
    ))


class DQNAgent:
    def __init__(self, env, config: Config = {}, id=""):
        self.env = env
        self.num_actions = self.env.action_space.n
        self.policy_net = DQN(self.env.observation_space.n,
                              self.num_actions)
        self.target_net = DQN(self.env.observation_space.n,
                              self.num_actions)
        if config:
            self.config = config
            self.target_net.load_state_dict(self.policy_net.state_dict())
            self.target_net.eval()  # target network in eval mode, not training
            self.target_update = config.target_update
            self.lr = config.lr
            self.lr_min = config.lr_min
            self.lr_decay = config.lr_decay
            self.current_lr = self.lr
            self.gamma = config.gamma
            self.loss_type = config.loss
            self.optimizer = torch.optim.Adam(
                params=self.policy_net.parameters(), lr=self.lr)  # TODO try other optimizer

            self.memory = ReplayMemory(config.memory_size)
            self.batch_size = config.batch_size

            self.eps_start = config.eps_start
            self.eps_min = config.eps_min
            self.eps_decay = config.eps_decay

            self.double_dqn = self.config.double_dqn

        self.device = torch.device("cpu")

        if id != "":
            self.id = str(id) + "-" + str(int(time.time()))
        else:
            self.id = str(int(time.time()))

    def log_decay(v_start, v_min, max_step, current_step):
        """
        Get the value of x at current_step using logarithme decay
        """
        rate = math.log(v_min / v_start) * -1 / max_step
        value = v_min + (v_start * (1 - rate) ** current_step)
        return value

    def _get_epsilon(self, episode):
        """
        Get epsilon used for greedy strategy,
        depending on current episode to apply decay
        """
        if episode >= self.config.learning_start:
            episode = episode - self.config.learning_start + 1
        else:
            episode = 1
        value = self.log_decay(
            self.eps_start, self.eps_min, self.eps_decay, episode)
        if value > self.eps_min:
            return value
        else:
            return self.eps_min

    def _update_learning_rate(self, episode):
        """
        Update the learning rate using logarithme decay
        """
        self.current_lr = self.log_decay(
            self.lr, self.lr_min, self.lr_decay, episode)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.current_lr

    def get_action(self, state):
        """
        Get an action depending on the state using policy network
        """
        with torch.no_grad():  # No grad because we use the model to select an action and not for training
            predicted = self.policy_net(
                torch.tensor([state], device=self.device))
            return predicted.argmax(dim=1).item()  # select max value index

    def _select_action(self, state, episode):
        """
        Select action during training,
        choosing beetwen explore or exploit depending of epsilon
        """
        epsilon = self._get_epsilon(episode)
        if epsilon > random.uniform(0, 1):
            return random.randrange(self.num_actions)  # explore
        else:
            return self.get_action(state)  # exploit

    def _memorize(self, state, action, new_state, reward, done):
        """
        Memorize env state in agent memory
        """
        experience = Experience(torch.tensor([state], device=self.device),
                                torch.tensor([action], device=self.device),
                                torch.tensor([new_state], device=self.device),
                                torch.tensor([reward], device=self.device),
                                torch.tensor([done], device=self.device, dtype=torch.bool))
        self.memory.push(experience)

    @staticmethod
    def extract_tensors(experiences):
        batch = Experience(*zip(*experiences))
        tensor_state = torch.cat(batch.state)
        tensor_action = torch.cat(batch.action)
        tensor_reward = torch.cat(batch.reward)
        tensor_next_state = torch.cat(batch.next_state)
        tensor_done = torch.cat(batch.done)
        return tensor_state, tensor_action, tensor_reward, tensor_next_state, tensor_done

    def _train_model(self):
        """
        Train model with a sample of replay memory
        """
        experiences = self.memory.sample(self.batch_size)
        states_b, actions_b, rewards_b, new_states_b, dones_b = self.extract_tensors(
            experiences)
        current_q_values = self.policy_net(states_b).gather(dim=1, index=actions_b.unsqueeze(
            1))  # send all states and actions pairs and get list of predicted qvalues

        if self.double_dqn:
            q_values_new = self.policy_net(new_states_b).detach()
            _, next_action_values = q_values_new.max(dim=1)
            q_target_new = self.target_net(new_states_b).detach()
            next_q_values = q_target_new.gather(
                dim=1, index=next_action_values.unsqueeze(1))
            next_q_values = next_q_values.squeeze()
            target_q_values = (~dones_b * next_q_values *
                               self.gamma) + rewards_b
        else:
            next_q_values = self.target_net(new_states_b).max(dim=1)[
                0]  # get target qvalues
            # target_q_values = (next_q_values * self.gamma) + \
            #    rewards_b  # compute expected qvalues
            target_q_values = (~dones_b * next_q_values *
                               self.gamma) + rewards_b

        if self.loss_type == 'mse':
            loss = nn.functional.mse_loss(
                current_q_values, target_q_values.unsqueeze(1))
        elif self.loss_type == 'huber':
            loss = nn.functional.huber_loss(
                current_q_values, target_q_values.unsqueeze(1))
        elif self.loss_type == 'mae':
            loss = nn.functional.smooth_l1_loss(
                current_q_values, target_q_values.unsqueeze(1))

        self.optimizer.zero_grad()  # prevent accumulating gradients during backprop
        loss.backward()  # compute the gradient of the loss with respect of weight and biases in the policy net
        for param in self.policy_net.parameters():
            # clips gradients computed during backpropagation to avoid explosion of gradients
            param.grad.data.clamp_(-1, 1)
        # update the weight and biases with the just previous gradients computed
        self.optimizer.step()

        return loss

    def fit(self, wandb_log=False, epochs=5000, save=True, max_step=100):
        """
        Function to train the model with
        previously set parameters
        """
        if wandb_log:
            run = wandb.init(
                project="T_AIA_902_msc2022_group-44", entity="hugoallegaert")
        scores = []
        try:
            episodes_reward = deque(maxlen=100)
            for episode in range(epochs):
                state = self.env.reset()
                episode_reward = 0
                loss = 0
                for step in range(max_step):
                    action = self._select_action(state, episode)
                    new_state, reward, done, _ = self.env.step(action)
                    self._memorize(state, action, new_state, reward, done)
                    state = new_state
                    episode_reward += reward

                    if episode >= self.config.learning_start and self.memory.get_current_len() >= self.batch_size:  # Train model
                        loss += self._train_model()

                    if done:
                        break
                if episode >= self.config.learning_start:
                    self._update_learning_rate(
                        episode - self.config.learning_start + 1)
                if episode >= self.config.learning_start and episode % self.target_update == 0:
                    self.target_net.load_state_dict(
                        self.policy_net.state_dict())  # update target network (we can do this in train_model function every x train for more stability)

                episodes_reward.append(episode_reward)
                scores.append(episode_reward)
                average_score = sum(episodes_reward) / len(episodes_reward)
                if episode % 100 == 0:
                    print(
                        f"Episode {episode + 100}\tAverage Score: {average_score:.2f}    ", end="\r")
                if wandb_log == True:
                    wandb.log({"reward": episode_reward, "duration": step,
                               "epsilon": self._get_epsilon(episode), "learning_rate": self.current_lr,
                               "loss": loss})
            if wandb_log:
                run.finish()
            if save:
                self._save()  # save trained model
        except KeyboardInterrupt:
            if save:
                self._save()
            if wandb_log:
                run.finish()
            print("Training has been interrupted")
        return average_score, scores

    def _save(self):
        """
        Save model weight into file
        """
        with open('./src/models/{0}.pt'.format(self.id), 'w') as f:
            torch.save(self.policy_net.state_dict(),
                       f"./src/models/{self.id}.pt")
            print('Model saved as: {0}.pt'.format(self.id))
        with open('./src/models/{0}.config.txt'.format(self.id), 'w') as f:
            t, p = self.evaluate(100)
            f.write(str(self.config).replace(',', ',\n'))
            f.write('\n---------------------------\n')
            f.write('Results after 100 episodes:\n')
            f.write(f"Average time steps per episode: {t}\n")
            f.write(f"Average penalties per episode: {p}")

    def load(self, file_path):
        self.policy_net.load_state_dict(torch.load(file_path))
        self.policy_net.eval()  # model in eval mode, not training

    def play_game(self, n_game):
        max_step = 100
        for episode in range(n_game):
            state = self.env.reset()
            game = "*** Game {0} ***\n".format(episode + 1)
            episode_reward = 0
            time.sleep(1)
            for _ in range(max_step):
                clear_output(wait=True)
                print(game, self.env.render('human'), end='\n')
                time.sleep(0.3)
                action = self.get_action(state)
                state, reward, done, _ = self.env.step(action)
                episode_reward += reward
                if done:
                    clear_output(wait=True)
                    print(game, self.env.render('human'), end='\n')
                    if reward == 1:
                        print("*** You Won {0} ***".format(episode_reward))
                        time.sleep(2)
                    else:
                        print("*** Your score {0} ***".format(episode_reward))
                        time.sleep(2)
                    clear_output(wait=True)
                    break

    def evaluate(self, n_game):
        total_steps, total_rewards, total_penalties = 0, 0, 0
        max_steps = 100
        for i in range(n_game):
            percent = round(((i + 1) / n_game) * 100)
            print("Loading {0}%".format(percent), end='\r')
            state = self.env.reset()
            penalties, rewards = 0, 0
            for step in range(max_steps):
                action = self.get_action(state)
                state, reward, done, _ = self.env.step(action)
                rewards += reward
                if reward == -10:
                    penalties += 1
                if done:
                    break
                if step >= max_steps:
                    penalties += 1
                    break
            total_penalties += penalties
            total_rewards += rewards
            total_steps += step
        print(f"Results after {n_game} episodes:")
        print(f"Average time steps per episode: {total_steps / n_game}")
        print(f"Average penalties per episode: {total_penalties / n_game}")
        print(f"Average rewards per episode: {total_rewards / n_game}")
        return total_steps / n_game, total_penalties / n_game
