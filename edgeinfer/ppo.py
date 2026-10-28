"""Teacher PPO structure: two 128-unit Tanh layers, summed joint log-probability,
GAE, clipped policy loss, separate critic and Adam optimizers. See docs/audit.md.
"""

from dataclasses import asdict
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical
from .config import Config, config_from_dict


class RunningNorm:
    """Welford moments. A zero-variance coordinate has unit scale by contract."""

    def __init__(self, shape):
        self.count = 0
        self.mean = np.zeros(shape, dtype=float)
        self.m2 = np.zeros(shape, dtype=float)

    def __call__(self, x, *, update=False):
        x = np.asarray(x, dtype=float)
        if update:
            self.count += 1
            delta = x - self.mean
            self.mean += delta / self.count
            self.m2 += delta * (x - self.mean)
        variance = self.m2 / self.count if self.count else np.zeros_like(self.mean)
        scale = np.sqrt(np.where(variance > 0, variance, 1.0))
        return (x - self.mean) / scale

    def state(self):
        return {"count": self.count, "mean": self.mean.tolist(), "m2": self.m2.tolist()}

    def restore(self, state):
        self.count = state["count"]
        self.mean = np.array(state["mean"], dtype=float)
        self.m2 = np.array(state["m2"], dtype=float)


class RewardScaler:
    def __init__(self, gamma):
        self.gamma = gamma
        self.discounted = 0.0
        self.norm = RunningNorm(())

    def reset(self):
        self.discounted = 0.0

    def __call__(self, reward):
        self.discounted = self.gamma * self.discounted + reward
        self.norm(self.discounted, update=True)
        variance = float(self.norm.m2 / self.norm.count)
        scale = variance**0.5 if variance > 0 else 1.0
        return reward / scale


def network(input_dim, hidden, output_dim, output_gain):
    result = nn.Sequential(nn.Linear(input_dim, hidden), nn.Tanh(), nn.Linear(hidden, hidden),
                           nn.Tanh(), nn.Linear(hidden, output_dim))
    for layer in (result[0], result[2]):
        nn.init.orthogonal_(layer.weight)
        nn.init.zeros_(layer.bias)
    nn.init.orthogonal_(result[4].weight, gain=output_gain)
    nn.init.zeros_(result[4].bias)
    return result


def gae_advantages(rewards, values, next_values, episode_ends, gamma, lam):
    """Time-limit truncations bootstrap value; GAE never crosses an episode reset."""
    deltas = np.asarray(rewards) + gamma * np.asarray(next_values) - np.asarray(values)
    advantages = np.empty_like(deltas)
    gae = 0.0
    for i in range(len(deltas) - 1, -1, -1):
        gae = deltas[i] + gamma * lam * (1 - episode_ends[i]) * gae
        advantages[i] = gae
    return advantages


class MaskedPPO:
    def __init__(self, config: Config, state_dim: int, seed: int, *, use_mask=True):
        torch.set_num_threads(1)
        torch.manual_seed(seed)
        self.config = config
        self.seed = seed
        self.use_mask = use_mask
        self.rng = np.random.default_rng(seed)
        c = config.ppo
        self.actor = network(state_dim, c.hidden, config.users * (config.nodes + 1), 0.01)
        self.critic = network(state_dim, c.hidden, 1, 1.0)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=c.actor_lr, eps=1e-5)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=c.critic_lr, eps=1e-5)
        self.norm = RunningNorm(state_dim)
        self.reward_scaler = RewardScaler(c.gamma)

    def distribution(self, states, masks):
        logits = self.actor(states).reshape(-1, self.config.users, self.config.nodes + 1)
        if not self.use_mask:
            masks = torch.ones_like(masks)
        return Categorical(logits=logits.masked_fill(~masks, -torch.inf))

    @torch.no_grad()
    def choose_normalized(self, state, mask, *, deterministic=False):
        states = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
        masks = torch.as_tensor(mask, dtype=torch.bool).reshape(1, self.config.users, -1)
        dist = self.distribution(states, masks)
        action = dist.probs.argmax(-1) if deterministic else dist.sample()
        return (action.numpy()[0], float(dist.log_prob(action).sum()), float(self.critic(states).item()))

    def act(self, observation, mask, *, deterministic=True):
        state = self.norm(observation)
        return self.choose_normalized(state, mask, deterministic=deterministic)[0]

    @torch.no_grad()
    def value(self, observation):
        state = torch.as_tensor(self.norm(observation), dtype=torch.float32).unsqueeze(0)
        return float(self.critic(state).item())

    def update(self, rollout, fraction):
        c = self.config.ppo
        adv = gae_advantages(rollout["rewards"], rollout["values"], rollout["next_values"],
                             rollout["ends"], c.gamma, c.gae_lambda)
        target = torch.tensor(adv + np.asarray(rollout["values"]), dtype=torch.float32)
        deviation = adv.std(ddof=1) if len(adv) > 1 else 0.0
        adv = (adv - adv.mean()) / (deviation if deviation > 0 else 1.0)
        states = torch.as_tensor(np.asarray(rollout["states"]), dtype=torch.float32)
        masks = torch.as_tensor(np.asarray(rollout["masks"]), dtype=torch.bool)
        actions = torch.as_tensor(np.asarray(rollout["actions"]), dtype=torch.int64)
        old_logprob = torch.tensor(rollout["logprobs"], dtype=torch.float32)
        advantages = torch.tensor(adv, dtype=torch.float32)
        losses = []
        for _ in range(c.epochs):
            permutation = self.rng.permutation(len(states))
            for start in range(0, len(states), c.minibatch):
                idx = permutation[start:start + c.minibatch]
                dist = self.distribution(states[idx], masks[idx])
                logprob = dist.log_prob(actions[idx]).sum(-1)
                ratio = (logprob - old_logprob[idx]).exp()
                surrogate = torch.minimum(ratio * advantages[idx], ratio.clamp(1 - c.clip, 1 + c.clip) * advantages[idx])
                actor_loss = -surrogate.mean() - c.entropy * dist.entropy().sum(-1).mean()
                self.actor_opt.zero_grad()
                actor_loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), c.max_grad_norm)
                self.actor_opt.step()
                critic_loss = (self.critic(states[idx]).flatten() - target[idx]).square().mean()
                self.critic_opt.zero_grad()
                critic_loss.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), c.max_grad_norm)
                self.critic_opt.step()
                losses.append((actor_loss.item(), critic_loss.item()))
        self.actor_opt.param_groups[0]["lr"] = c.actor_lr * fraction
        self.critic_opt.param_groups[0]["lr"] = c.critic_lr * fraction
        return np.mean(losses, axis=0).tolist()

    def save(self, path: str | Path):
        torch.save({"config": asdict(self.config), "seed": self.seed,
                    "use_mask": self.use_mask,
                    "state_dim": int(self.actor[0].in_features),
                    "actor": self.actor.state_dict(), "critic": self.critic.state_dict(),
                    "normalization": self.norm.state()}, path)

    @classmethod
    def load(cls, path: str | Path):
        data = torch.load(path, map_location="cpu", weights_only=True)
        # v0.1 checkpoints are masked policies and predate this explicit field.
        agent = cls(config_from_dict(data["config"]), data["state_dim"], data["seed"],
                    use_mask=data.get("use_mask", True))
        agent.actor.load_state_dict(data["actor"])
        agent.critic.load_state_dict(data["critic"])
        agent.norm.restore(data["normalization"])
        return agent

