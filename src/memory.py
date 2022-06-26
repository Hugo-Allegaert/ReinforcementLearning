import random
from collections import namedtuple

Experience = namedtuple(
    'Experience', ('state', 'action', 'next_state', 'reward', 'done'))


class ReplayMemory:
    def __init__(self, max_size):
        self.max_size = max_size
        self.__memory = []
        self.push_count = 0

    def push(self, experience):
        if len(self.__memory) < self.max_size:
            self.__memory.append(experience)
        else:
            self.__memory[self.push_count % self.max_size] = experience
        self.push_count += 1

    def sample(self, batch_size):
        return random.sample(self.__memory, batch_size)

    def get_current_len(self):
        return len(self.__memory)

    def get_memory(self):
        return self.__memory
