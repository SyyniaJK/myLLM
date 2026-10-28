"""Basic scheduling comparators; all see only the same observation and mask."""

import numpy as np
from .experiment import ChannelPolicy, RandomPolicy
from .physics import batch_cost


class BalancedPolicy:
    def __init__(self, config, seed):
        self.config = config

    def act(self, observation, mask, *, deterministic=True):
        c = self.config
        active = observation[:c.users] > 0
        gains = observation[3 * c.users:3 * c.users + c.users * c.nodes].reshape(c.users, c.nodes)
        load = np.zeros(c.nodes, dtype=int)
        action = np.zeros(c.users, dtype=int)
        for user in np.flatnonzero(active):
            available = np.flatnonzero(mask[user, 1:])
            if available.size:
                node = min(available, key=lambda j: (load[j], -gains[user, j], j))
                action[user] = node + 1
                load[node] += 1
        return action


class CapGreedyPolicy:
    """Longest declared cap first; select the lowest resulting peak-memory batch.

    This is a platform baseline, not the paper's cited weighted Heuristic.
    """

    def __init__(self, config, seed):
        self.config = config

    def act(self, observation, mask, *, deterministic=True):
        c = self.config
        inputs = observation[:c.users]
        caps = observation[c.users:2 * c.users]
        gains = observation[3 * c.users:3 * c.users + c.users * c.nodes].reshape(c.users, c.nodes)
        batches = [[] for _ in range(c.nodes)]
        action = np.zeros(c.users, dtype=int)
        active = sorted(np.flatnonzero(inputs > 0), key=lambda u: (-caps[u], -inputs[u], u))
        for user in active:
            candidates = []
            for node in np.flatnonzero(mask[user, 1:]):
                group = batches[node] + [user]
                cost = batch_cost(c.model, inputs[group], caps[group], c.flops_s)
                if cost.peak_bytes <= c.capacity_bytes:
                    candidates.append((cost.peak_bytes, -gains[user, node], int(node)))
            if candidates:
                node = min(candidates)[2]
                action[user] = node + 1
                batches[node].append(user)
        return action


def random_masked(config, seed):
    return RandomPolicy(seed)


def strongest_channel(config, seed):
    return ChannelPolicy(config)


FACTORIES = {"random_masked": random_masked, "strongest_channel": strongest_channel,
             "balanced": BalancedPolicy, "cap_greedy": CapGreedyPolicy}
