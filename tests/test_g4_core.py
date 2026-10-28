from dataclasses import replace
import numpy as np
import torch
from edgeinfer.benchmark import evaluate_episode, metric_row, scientific_metrics
from edgeinfer.config import Device
from edgeinfer.environment import EdgeEnv
from edgeinfer.experiment import RandomPolicy, train
from edgeinfer.policies import BalancedPolicy, CapGreedyPolicy
from edgeinfer.ppo import MaskedPPO
from edgeinfer.scenarios import Scenario, FixedWorkload
from edgeinfer.workload import Request


def test_unmasked_distribution_uses_same_likelihood_at_sampling_and_update(tiny_config):
    agent = MaskedPPO(tiny_config, 7, 1, use_mask=False)
    states = torch.zeros(2, 7)
    mask = torch.zeros(2, 2, 3, dtype=torch.bool)
    mask[:, :, 0] = True
    masked_argument = agent.distribution(states, mask)
    unmasked_argument = agent.distribution(states, torch.ones_like(mask))
    torch.testing.assert_close(masked_argument.probs, unmasked_argument.probs)
    assert (masked_argument.probs > 0).all()


def test_unmasked_training_checkpoint_and_repeatability(tiny_config, tmp_path):
    config = replace(tiny_config, ppo=replace(tiny_config.ppo, hidden=16, rollout_steps=16, minibatch=8, epochs=2))
    a, curve_a, _ = train(config, 3, 32, tmp_path / "a", eval_every=16, eval_episodes=1, algorithm="tp_ppo")
    b, curve_b, _ = train(config, 3, 32, tmp_path / "b", eval_every=16, eval_episodes=1, algorithm="tp_ppo")
    loaded = MaskedPPO.load(tmp_path / "a" / "policy.pt")
    assert not a.use_mask and not loaded.use_mask
    assert curve_a == curve_b
    assert all(torch.equal(x, y) for x, y in zip(a.actor.parameters(), b.actor.parameters()))


def test_scenario_roundtrip_and_legacy_closed_loop_are_exact(tiny_config, tmp_path):
    scenario = Scenario.generate(tiny_config, 9)
    scenario.save(tmp_path)
    loaded = Scenario.load(tmp_path)
    assert scenario.config == loaded.config and scenario.offers == loaded.offers
    np.testing.assert_array_equal(scenario.up, loaded.up)
    np.testing.assert_array_equal(scenario.down, loaded.down)
    old, new = EdgeEnv(tiny_config), EdgeEnv(tiny_config)
    o0, i0 = old.reset(seed=9)
    o1, i1 = new.reset(seed=9, options={"scenario": loaded, "workload_mode": "closed_loop"})
    p0, p1 = RandomPolicy(1), RandomPolicy(1)
    for _ in range(tiny_config.slots):
        np.testing.assert_array_equal(o0, o1)
        a0, a1 = p0.act(o0, i0["action_mask"]), p1.act(o1, i1["action_mask"])
        np.testing.assert_array_equal(a0, a1)
        o0, r0, _, _, i0 = old.step(a0)
        o1, r1, _, _, i1 = new.step(a1)
        assert r0 == r1
    assert old.records == new.records and old.summary() == new.summary()


def test_fixed_offers_are_identical_for_different_actions(tiny_config):
    scenario = Scenario.generate(replace(tiny_config, mean_think_s=0.1), 8)
    class Reject:
        def act(self, observation, mask, *, deterministic=True):
            return np.zeros(tiny_config.users, dtype=int)
    first, _, _, _ = evaluate_episode(scenario, "fixed", Reject())
    second, _, _, _ = evaluate_episode(scenario, "fixed", BalancedPolicy(scenario.config, 1))
    fields = Request.__dataclass_fields__
    signature = lambda env: sorted(tuple(r[k] for k in fields) for r in env.records)
    assert signature(first) == signature(second)
    assert first.summary()["goodput"] < second.summary()["goodput"]


def test_fixed_mode_allows_explicitly_overlapping_offers(tiny_config):
    config = replace(tiny_config, device=Device("slow", 0.01, 1e-10))
    scenario = replace(Scenario.generate(config, 7), offers=(
        Request(0, 0, 0.1, 2, 2, 2, 100), Request(0, 1, 1.1, 2, 2, 2, 100)))
    env = EdgeEnv(config)
    env.reset(seed=7, options={"scenario": scenario, "workload_mode": "fixed"})
    env.step(np.array([1, 0]))
    assert env.records[0]["response_s"] > 2
    assert 0 in env.current_requests
    env.step(np.array([2, 0]))
    assert len(env.records) == 2 and env.records[1]["node"] == 1


def test_cap_greedy_uses_declared_caps_not_realized_output(tiny_config):
    env = EdgeEnv(tiny_config)
    obs, info = env.reset(seed=7)
    p = CapGreedyPolicy(tiny_config, 1)
    action = p.act(obs, info["action_mask"])
    assert np.all(info["action_mask"][np.arange(tiny_config.users), action])


def test_fixed_scenario_energy_observer_and_metrics_boundary(tiny_config):
    config = replace(tiny_config, slots=1, device=Device("slow", 0.01, 1e-10))
    scenario = replace(Scenario.generate(config, 7), offers=(Request(0, 0, 0.1, 2, 2, 2, 100),))
    a, aa, ar, timing = evaluate_episode(scenario, "fixed", BalancedPolicy(config, 1))
    b, ba, br, _ = evaluate_episode(scenario, "fixed", BalancedPolicy(config, 1), energy=False)
    np.testing.assert_array_equal(aa, ba)
    np.testing.assert_array_equal(ar, br)
    assert a.records == b.records
    row = metric_row(a, timing)
    assert row["goodput_window"] == 0 and row["goodput_cohort"] == 1
    assert row["e2e_mean_s"] is None and row["energy_j"] > 0
    changed_clock = dict(row, policy_decision_ms_mean=999)
    assert scientific_metrics(row) == scientific_metrics(changed_clock)
