import unittest

from src.memory import ReplayMemory, Experience


class TestMemory(unittest.TestCase):

    def test_push(self):
        memory = ReplayMemory(2)
        experience1 = Experience(46, 1, 46, -1, False)
        experience2 = Experience(58, 0, 58, -5, False)
        experience3 = Experience(72, 1, 72, -10, True)

        memory.push(experience1)
        memory.push(experience2)
        memory.push(experience3)

        self.assertEqual(memory.push_count, 3, "Expected push to be the number of experiences added")
        self.assertEqual(memory.get_memory()[0], experience3, "Expected first memory to be the first experience")
        self.assertEqual(memory.get_memory()[1], experience2, "Expected second memory to be the second experience")
        self.assertEqual(memory.get_current_len(), 2, "Expected number of memories to be equal to the max-size")

    def test_sample(self):
        memory = ReplayMemory(3)
        experience1 = Experience(55, 1, 55, -1, False)
        experience2 = Experience(66, 0, 66, -5, False)
        experience3 = Experience(77, 1, 77, -10, True)

        memory.push(experience1)
        memory.push(experience2)
        memory.push(experience3)

        self.assertIn(memory.sample(1)[0], [experience1, experience2, experience3],
                      "Expected memory sample to be one of the experiences added")


if __name__ == '__main__':
    unittest.main()
