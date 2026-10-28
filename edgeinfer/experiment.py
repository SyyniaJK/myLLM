"""Training/evaluation share the same environment and physical accounting."""

import csv
from dataclasses import asdict
import importlib.metadata
import json
from pathlib import Path
import platform
import time
from typing import Protocol
import numpy as np
from .config import Config
from .environment import EdgeEnv
from .ppo import MaskedPPO


class Policy(Protocol):
    def act(self, observation: np.ndarray, mask: np.ndarray, *, deterministic: bool = True) -> np.ndarray: ...


class RandomPolicy:
    def __init__(self, seed):
        self.rng = np.random.default_rng(seed)

    def act(self, observation, mask, *, deterministic=True):
        return np.array([self.rng.choice(np.flatnonzero(row)) for row in mask], dtype=int)


class ChannelPolicy:
    """Diagnostic strongest-uplink rule; not the paper's cited Heuristic baseline."""

    def __init__(self, config):
        self.config = config

    def act(self, observation, mask, *, deterministic=True):
        c = self.config
        gains = observation[3 * c.users:3 * c.users + c.users * c.nodes].reshape(c.users, c.nodes)
        scores = np.column_stack((np.full(c.users, -1.0), gains))
        return np.where(mask, scores, -np.inf).argmax(axis=1)


def write_csv(path, rows):
    rows = list(rows)
    assert rows, "A CSV result has at least one row."
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def evaluate(config, policy, seeds, *, energy=True, traces_dir=None):
    env = EdgeEnv(config, energy=energy)
    rows = []
    for seed in seeds:
        obs, info = env.reset(seed=int(seed))
        for _ in range(config.slots):
            action = policy.act(obs, info["action_mask"], deterministic=True)
            obs, _, _, _, info = env.step(action)
        summary = {"seed": int(seed), **env.summary()}
        rows.append(summary)
        if traces_dir is not None:
            traces_dir = Path(traces_dir)
            traces_dir.mkdir(parents=True, exist_ok=True)
            write_csv(traces_dir / f"requests_{seed}.csv", env.records)
            write_json(traces_dir / f"batches_{seed}.json", env.batch_records)
            write_json(traces_dir / f"power_intervals_{seed}.json", env.ledger.rows())
    return rows


def mean_metrics(rows):
    keys = ("reward", "goodput", "goodput_ratio", "throughput", "inference_completed", "requests", "oom_requests",
            "rejected", "downlink_dropped", "gpu_ttft_s", "gpu_tpot_s", "e2e_s", "pending_at_horizon")
    result = {}
    for key in keys:
        values = [row[key] for row in rows if row[key] is not None]
        result[key] = float(np.mean(values)) if values else None
    result["energy_window_j"] = float(np.mean([x["energy_window"]["total_j"] for x in rows])) if rows[0]["energy_window"] is not None else None
    return result


def runtime_versions():
    return {"python": platform.python_version(),
            **{name: importlib.metadata.version(name) for name in ("numpy", "torch", "gymnasium", "matplotlib")}}


def train(config: Config, seed: int, steps: int, output, *, eval_every=4096, eval_episodes=5,
          algorithm="tp_mppo"):
    assert steps > 0 and eval_every > 0 and eval_episodes > 0
    assert algorithm in ("tp_mppo", "tp_ppo")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "run.json", {"config": asdict(config), "seed": seed, "training_steps": steps,
                                   "runtime": runtime_versions(), "algorithm": algorithm})
    env = EdgeEnv(config, energy=False)
    obs, info = env.reset(seed=seed)
    agent = MaskedPPO(config, len(obs), seed, use_mask=algorithm == "tp_mppo")
    eval_seeds = list(range(10_000, 10_000 + eval_episodes))
    curve = [{"step": 0, "episode_equivalent": 0.0,
              **mean_metrics(evaluate(config, agent, eval_seeds))}]
    episode_rewards = []
    episode_reward = 0.0
    last_eval = 0
    start_time = time.perf_counter()
    for start in range(0, steps, config.ppo.rollout_steps):
        rollout = {key: [] for key in ("states", "masks", "actions", "logprobs", "rewards", "values", "next_values", "ends")}
        stop = min(start + config.ppo.rollout_steps, steps)
        for _ in range(start, stop):
            state = agent.norm(obs, update=True)
            mask = info["action_mask"]
            action, logprob, value = agent.choose_normalized(state, mask)
            obs, reward, terminated, truncated, info = env.step(action)
            next_value = 0.0 if terminated else agent.value(obs)
            scaled_reward = agent.reward_scaler(reward) if config.ppo.reward_scaling else reward
            values = (state, mask, action, logprob, scaled_reward, value, next_value, terminated or truncated)
            for key, value_item in zip(rollout, values):
                rollout[key].append(value_item)
            episode_reward += reward
            if terminated or truncated:
                episode_rewards.append({"episode": len(episode_rewards) + 1, "reward": episode_reward})
                episode_reward = 0.0
                agent.reward_scaler.reset()
                obs, info = env.reset()
        agent.update(rollout, 1 - stop / steps)
        if stop - last_eval >= eval_every or stop == steps:
            row = {"step": stop, "episode_equivalent": stop / config.slots,
                   **mean_metrics(evaluate(config, agent, eval_seeds))}
            curve.append(row)
            last_eval = stop
            write_csv(output / "learning_curve.csv", curve)
            print(f"{output.parent.name}/{output.name} step={stop}/{steps} reward={row['reward']:.3f} goodput={row['goodput_ratio']:.3%}", flush=True)
    agent.save(output / "policy.pt")
    write_csv(output / "learning_curve.csv", curve)
    write_csv(output / "training_episodes.csv", episode_rewards)
    test = evaluate(config, agent, range(20_000, 20_000 + eval_episodes), traces_dir=output / "test_traces")
    write_json(output / "test_episodes.json", test)
    write_json(output / "summary.json", {"metrics": mean_metrics(test), "wall_s": time.perf_counter() - start_time,
                                       "initial_validation": curve[0], "final_validation": curve[-1]})
    return agent, curve, test

