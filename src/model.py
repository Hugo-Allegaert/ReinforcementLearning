import torch.nn as nn
import torch.nn.functional as F


class DQN(nn.Module):
    def __init__(self, n_state, n_action):
        super().__init__()
        self.emb = nn.Embedding(n_state, 64)
        self.ll1 = nn.Linear(in_features=64, out_features=64)
        self.ll2 = nn.Linear(in_features=64, out_features=64)
        self.out_features = nn.Linear(in_features=64, out_features=n_action)

    def forward(self, input_t):
        input_t = F.relu(self.ll1(self.emb(input_t)))
        input_t = F.relu(self.ll2(input_t))
        input_t = self.out_features(input_t)
        return input_t
