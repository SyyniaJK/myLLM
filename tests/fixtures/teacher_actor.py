"""Frozen Actor reference from the user-provided ppo_discrete.py.

Retained only for the numerical parity test; not imported by the simulator.
"""
import torch
from torch import nn


def orthogonal_init(layer, gain=1.0):
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.constant_(layer.bias, 0)


class Actor(nn.Module):
    def __init__(self, args):
        super(Actor, self).__init__()
        self.num_users = args.num_users
        self.action_dim_per_user = args.action_dim_per_user
        self.fc1 = nn.Linear(args.state_dim, args.hidden_width)
        self.fc2 = nn.Linear(args.hidden_width, args.hidden_width)
        self.fc3 = nn.Linear(args.hidden_width, self.num_users * self.action_dim_per_user)
        self.activate_func = [nn.ReLU(), nn.Tanh()][args.use_tanh]
        if args.use_orthogonal_init:
            orthogonal_init(self.fc1)
            orthogonal_init(self.fc2)
            orthogonal_init(self.fc3, gain=0.01)

    def forward(self, s, mask=None):
        s = self.activate_func(self.fc1(s))
        s = self.activate_func(self.fc2(s))
        logits = self.fc3(s)
        logits = logits.view(-1, self.num_users, self.action_dim_per_user)
        if mask is not None:
            logits[~mask] = -1e8
        a_prob = torch.softmax(logits, dim=-1)
        return a_prob
