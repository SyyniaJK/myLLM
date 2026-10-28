"""Closed-loop users: one outstanding request, then independent exponential thinking."""

from dataclasses import dataclass
import numpy as np
from .config import Config


@dataclass(frozen=True)
class Request:
    user: int
    sequence: int
    arrival_s: float
    input_tokens: int
    output_cap: int
    output_tokens: int
    slo_s: float


class ClosedLoopWorkload:
    def __init__(self, config: Config, seed: int):
        self.config = config
        self.request_rng = [np.random.default_rng(np.random.SeedSequence([seed, 20, u])) for u in range(config.users)]
        self.think_rng = [np.random.default_rng(np.random.SeedSequence([seed, 21, u])) for u in range(config.users)]
        self.next_arrival = np.array([r.exponential(config.mean_think_s) for r in self.think_rng])
        self.sequence = np.zeros(config.users, dtype=int)

    def arrivals(self, start_s: float, end_s: float):
        ready = np.flatnonzero(self.next_arrival < end_s)
        assert np.all(self.next_arrival[ready] >= start_s), "A consumed arrival must never remain pending."
        requests = {}
        c = self.config
        for u in ready:
            rng = self.request_rng[u]
            kind = int(rng.integers(len(c.input_tokens)))
            cap = int(rng.choice(c.output_caps))
            output = int(cap * rng.choice(c.output_fractions))
            requests[int(u)] = Request(int(u), int(self.sequence[u]), float(self.next_arrival[u]),
                                       c.input_tokens[kind], cap, output, c.slo_s[kind])
            self.sequence[u] += 1
            self.next_arrival[u] = np.inf
        return requests

    def finish(self, user: int, response_s: float):
        assert np.isinf(self.next_arrival[user]), "A user has exactly one outstanding request."
        self.next_arrival[user] = response_s + self.think_rng[user].exponential(self.config.mean_think_s)

