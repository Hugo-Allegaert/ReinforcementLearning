from torch.distributions.categorical import Categorical
import torch.nn.functional as F
import torch.nn as nn
import torch


class DQN(nn.Module):
    def __init__(self, n_state, n_action, softmax=False):
        super().__init__()
        self.softmax = softmax
        self.emb = nn.Embedding(n_state, 64)
        self.ll1 = nn.Linear(in_features=64, out_features=64)
        self.ll2 = nn.Linear(in_features=64, out_features=64)
        self.out_features = nn.Linear(in_features=64, out_features=n_action)
        if softmax == True:
            self.llsoftmax = nn.Softmax(dim=-1)

    def forward(self, input_t):
        # input_t = self.emb(input_t.long())
        input_t = F.relu(self.ll1(self.emb(input_t)))
        input_t = F.relu(self.ll2(input_t))
        input_t = self.out_features(input_t)
        if self.softmax == True:
            input_t = self.llsoftmax(input_t)
            input_t = Categorical(input_t)
        return input_t
