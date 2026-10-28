from dataclasses import replace
import numpy as np
import torch
import importlib.util
from pathlib import Path
from types import SimpleNamespace
from edgeinfer.environment import EdgeEnv
from edgeinfer.ppo import MaskedPPO, RunningNorm, gae_advantages
from edgeinfer.experiment import train


def test_gae_bootstraps_truncation_but_does_not_cross_reset():
    result = gae_advantages([1.0, 2.0], [0.5, 0.25], [0.75, 0.0], [True, True], 0.9, 0.95)
    np.testing.assert_allclose(result, [1 + 0.9 * 0.75 - 0.5, 2 - 0.25])


def test_welford_zero_variance_has_unit_scale():
    norm = RunningNorm(2)
    norm([0, 1], update=True)
    result = norm([0, 1], update=True)
    np.testing.assert_array_equal(result, [0, 0])
    assert np.isfinite(result).all()


def test_joint_logprob_and_invalid_action_probability(tiny_config):
    env = EdgeEnv(tiny_config)
    obs, _ = env.reset(seed=1)
    agent = MaskedPPO(tiny_config, len(obs), 1)
    mask = torch.tensor([[[True, False, True], [True, False, False]]])
    dist = agent.distribution(torch.zeros(1, len(obs)), mask)
    assert dist.probs[0, 0, 1] == 0
    assert dist.probs[0, 1, 0] == 1
    action = torch.tensor([[2, 0]])
    assert dist.log_prob(action).sum().item() == torch.log(dist.probs[0, 0, 2]).item()


def test_checkpoint_round_trip(tiny_config, tmp_path):
    env = EdgeEnv(tiny_config)
    obs, info = env.reset(seed=1)
    agent = MaskedPPO(tiny_config, len(obs), 1)
    agent.norm(obs, update=True)
    agent.save(tmp_path / "policy.pt")
    loaded = MaskedPPO.load(tmp_path / "policy.pt")
    np.testing.assert_array_equal(agent.act(obs, info["action_mask"]), loaded.act(obs, info["action_mask"]))


def test_training_updates_policy_and_is_seed_reproducible(tiny_config, tmp_path):
    config = replace(tiny_config, ppo=replace(tiny_config.ppo, hidden=16, rollout_steps=16, minibatch=8, epochs=2))
    env = EdgeEnv(config)
    obs, _ = env.reset(seed=7)
    initial = MaskedPPO(config, len(obs), 7)
    a, curve_a, _ = train(config, 7, 32, tmp_path / "a", eval_every=16, eval_episodes=1)
    b, curve_b, _ = train(config, 7, 32, tmp_path / "b", eval_every=16, eval_episodes=1)
    assert curve_a == curve_b
    assert any(not torch.equal(x, y) for x, y in zip(initial.actor.parameters(), a.actor.parameters()))
    assert all(torch.equal(x, y) for x, y in zip(a.actor.parameters(), b.actor.parameters()))


def test_actor_initialization_and_mask_match_teacher_code(tiny_config):
    source = Path(__file__).parent / "fixtures" / "teacher_actor.py"
    spec = importlib.util.spec_from_file_location("teacher_ppo", source)
    teacher = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(teacher)
    args = SimpleNamespace(num_users=2, action_dim_per_user=3, state_dim=7,
                           hidden_width=128, use_tanh=1, use_orthogonal_init=True)
    torch.manual_seed(23)
    original = teacher.Actor(args)
    rebuilt = MaskedPPO(tiny_config, 7, 23)
    state = torch.arange(7, dtype=torch.float32).reshape(1, -1)
    mask = torch.tensor([[[True, False, True], [True, True, True]]])
    torch.testing.assert_close(original(state, mask), rebuilt.distribution(state, mask).probs)
