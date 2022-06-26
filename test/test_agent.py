import unittest
import gym
import torch

from src.agent import DQNAgent, Config
from src.memory import Experience

env_name = "Taxi-v3"
env = gym.make(env_name)

config = Config(target_update=10,
                lr=0.01,
                lr_min=0.001,
                lr_decay=1000,
                gamma=0.99,
                loss='mse',
                memory_size=50000,
                batch_size=128,
                eps_start=1,
                eps_min=0.01,
                eps_decay=3000,
                learning_start=200,
                double_dqn=True)


class TestMemory(unittest.TestCase):

    def test_log_decay(self):
        agent = DQNAgent(env, config, id="test_config")
        self.assertEqual(agent.log_decay(1, 0.01, 3000, 0), 1.01,
                         "Expected to log a big logarithm decay at the first step")
        self.assertEqual(agent.log_decay(1, 0.01, 3000, 3000), 0.01996468032008466,
                         "Expected to log a small logarithm decay at the last step")

    def test_extract_tensors(self):
        agent = DQNAgent(env, config, id="test_config")

        experience1 = Experience(
            torch.tensor([46]),
            torch.tensor([1]),
            torch.tensor([47]),
            torch.tensor([-1]),
            torch.tensor([False])
        )
        experience2 = Experience(
            torch.tensor([58]),
            torch.tensor([0]),
            torch.tensor([59]),
            torch.tensor([-5]),
            torch.tensor([False])
        )
        experience3 = Experience(
            torch.tensor([72]),
            torch.tensor([1]),
            torch.tensor([73]),
            torch.tensor([-10]),
            torch.tensor([True])
        )
        experiences = [experience1, experience2, experience3]

        test_states, test_actions, test_rewards, test_new_states, test_dones = agent.extract_tensors(experiences)

        self.assertTrue(torch.equal(test_states, torch.tensor([46, 58, 72])),
                        "Expected to aggregate the state's tensors")
        self.assertTrue(torch.equal(test_actions, torch.tensor([1, 0, 1])),
                        "Expected to aggregate the action's tensors")
        self.assertTrue(torch.equal(test_rewards, torch.tensor([-1, -5, -10])),
                        "Expected to aggregate the reward's tensors")
        self.assertTrue(torch.equal(test_new_states, torch.tensor([47, 59, 73])),
                        "Expected to aggregate the new state's tensors")
        self.assertTrue(torch.equal(test_dones, torch.tensor([False, False, True])),
                        "Expected to aggregate the done's tensors")

    def test_evaluate(self):
        agent = DQNAgent(env, config, id="test_config")

        test_t, test_p = agent.evaluate(1000)
        self.assertEqual(test_t, 99, "Expected the number of steps played in a game to be stable")
        # self.assertEqual(test_p, 5) results are too random to be tested


if __name__ == '__main__':
    unittest.main()
