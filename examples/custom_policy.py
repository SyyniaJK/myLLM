"""Example plug-in: --custom cyclic=examples.custom_policy:build --methods cyclic."""

import numpy as np
from edgeinfer.allocation import EqualBandwidth


class CyclicPolicy:
    def __init__(self, config, seed):
        self.config = config
        self.next_node = seed % config.nodes

    def act(self, observation, mask, *, deterministic=True):
        c = self.config
        action = np.zeros(c.users, dtype=int)
        for user in np.flatnonzero(observation[:c.users] > 0):
            order = (self.next_node + np.arange(c.nodes)) % c.nodes
            available = order[mask[user, order + 1]]
            if available.size:
                node = int(available[0])
                action[user] = node + 1
                self.next_node = (node + 1) % c.nodes
        return action


def build(config, seed):
    return CyclicPolicy(config, seed), 0


def equal_resources(config):
    return EqualBandwidth()
