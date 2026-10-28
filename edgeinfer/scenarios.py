"""Saved exogenous inputs for paired evaluation, without file digests."""

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import numpy as np
from .config import Config, config_from_dict
from .physics import channel_trace
from .workload import ClosedLoopWorkload, Request


def fixed_offers(config: Config, episode_seed: int):
    """At most one offer/user/slot; independent of every scheduling outcome."""
    result = []
    for user in range(config.users):
        rng = np.random.default_rng(np.random.SeedSequence([episode_seed, 41, user]))
        sequence = 0
        for slot in range(config.slots):
            offset = rng.exponential(config.mean_think_s)
            if offset < config.slot_s:
                kind = int(rng.integers(len(config.input_tokens)))
                cap = int(rng.choice(config.output_caps))
                output = int(cap * rng.choice(config.output_fractions))
                result.append(Request(user, sequence, slot * config.slot_s + float(offset),
                                      config.input_tokens[kind], cap, output, config.slo_s[kind]))
                sequence += 1
    return tuple(sorted(result, key=lambda r: (r.arrival_s, r.user)))


class FixedWorkload:
    """Open-loop offers can overlap earlier requests; no cross-slot AP queue."""

    def __init__(self, config, requests):
        self.config = config
        self.by_slot = [dict() for _ in range(config.slots + 1)]
        for request in requests:
            assert 0 <= request.user < config.users
            assert 0 <= request.arrival_s < config.horizon_s
            slot = int(request.arrival_s // config.slot_s)
            assert request.user not in self.by_slot[slot], "At most one offer per user per slot."
            self.by_slot[slot][request.user] = request

    def arrivals(self, start_s, end_s):
        slot = int(round(start_s / self.config.slot_s))
        return dict(self.by_slot[slot])

    def finish(self, user, response_s):
        # Exogenous arrivals do not depend on success, rejection or completion.
        pass


@dataclass(frozen=True)
class Scenario:
    config: Config
    seed: int
    episode_seed: int
    up: np.ndarray
    down: np.ndarray
    offers: tuple[Request, ...]

    @classmethod
    def generate(cls, config, seed):
        episode_seed = int(np.random.default_rng(seed).integers(0, 2**31))
        up, down = channel_trace(config, episode_seed)
        return cls(config, seed, episode_seed, up, down, fixed_offers(config, episode_seed))

    def workload(self, mode):
        assert mode in ("closed_loop", "fixed")
        if mode == "fixed":
            return FixedWorkload(self.config, self.offers)
        return ClosedLoopWorkload(self.config, self.episode_seed)

    def save(self, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        data = {"format": 1, "config": asdict(self.config), "seed": self.seed,
                "episode_seed": self.episode_seed, "offers": [asdict(x) for x in self.offers]}
        (directory / "scenario.json").write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        np.savez_compressed(directory / "channels.npz", up=self.up, down=self.down)

    @classmethod
    def load(cls, directory):
        directory = Path(directory)
        data = json.loads((directory / "scenario.json").read_text(encoding="utf-8"))
        assert data["format"] == 1
        config = config_from_dict(data["config"])
        with np.load(directory / "channels.npz", allow_pickle=False) as channels:
            up, down = channels["up"], channels["down"]
        shape = (config.slots + 1, config.users, config.nodes)
        assert up.shape == down.shape == shape
        assert np.all(up > 0) and np.all(down > 0)
        return cls(config, data["seed"], data["episode_seed"], up, down,
                   tuple(Request(**r) for r in data["offers"]))
