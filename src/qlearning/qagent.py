import random
import numpy as np


class QAgent():
    def __init__(self, env):
        self.env = env
        self.action_space = env.action_space
        self.q_table = np.zeros((env.observation_space.n, env.action_space.n))
        self.learning_rate = 0.1
        self.discount_rate = 0.99
        self.exploration_rate = 1
        self.max_exploration = 1
        self.min_exploration = 0.01
        self.exploration_decay_rate = 0.001

    def get_action(self, state):
        greedy = random.uniform(0, 1)
        if greedy > self.exploration_rate:
            action = np.argmax(self.q_table[state, :])
        else:
            action = self.action_space.sample()
        return action

    def update_q_table(self, state, action, reward, new_state):
        self.q_table[state, action] = (1 - self.learning_rate) * self.q_table[state, action] + \
            (self.learning_rate * (reward +
             (self.discount_rate * np.max(self.q_table[new_state, :]))))
        return 1

    def decay_exploration_rate(self, episode):
        self.exploration_rate = self.min_exploration + \
            (self.max_exploration - self.min_exploration) * \
            np.exp(-self.exploration_decay_rate * episode)
        return 1

    def fit(self, epochs=5000, max_step=100):
        scores = []
        for episode in range(epochs):
            state = self.env.reset()
            reward_episode = 0
            for _ in range(max_step):
                action = self.get_action(state)
                new_state, reward, done, _ = self.env.step(action)
                self.update_q_table(state, action, reward, new_state)
                state = new_state
                reward_episode += reward
                if done:
                    break
            self.decay_exploration_rate(episode)
            scores.append(reward_episode)
            if episode % 100 == 0:
                print(
                    f"Episode {episode + 100}\tAverage Score: {sum(scores[-100:])/100:.2f}       ", end="\r")
        return scores
