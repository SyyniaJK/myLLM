import numpy as np
import pytest
from edgeinfer.energy import EnergyLedger, integrate_power_samples
from edgeinfer.environment import EdgeEnv
from edgeinfer.experiment import RandomPolicy


def test_known_power_integral():
    assert integrate_power_samples([0, 1, 3], [10, 20, 20]) == 55


def test_idle_and_increment_are_not_double_counted(tiny_config):
    ledger = EnergyLedger(tiny_config)
    idle = tiny_config.energy.idle_w_per_gpu
    active = tiny_config.energy.prefill_w_per_gpu
    ledger.record("gpu_prefill_increment_j", 1, 3, active - idle, 0)
    summary = ledger.summary(0, 4)
    expected = (tiny_config.nodes * 4 - 2) * idle + 2 * active
    assert summary["total_j"] == expected


def test_interval_clipping_and_component_sum(tiny_config):
    ledger = EnergyLedger(tiny_config)
    ledger.record("uplink_j", 0.5, 2.5, 2, 0)
    ledger.record("downlink_j", 2, 4, 3, 0)
    summary = ledger.summary(0, 3)
    assert summary["uplink_j"] == 4
    assert summary["downlink_j"] == 3
    assert summary["total_j"] == sum(v for k, v in summary.items() if k != "total_j")
    assert ledger.summary(0, 0)["total_j"] == 0


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_energy_observer_does_not_change_any_decision_or_transition(tiny_config, seed):
    envs = [EdgeEnv(tiny_config, energy=flag) for flag in (False, True)]
    policies = [RandomPolicy(15), RandomPolicy(15)]
    states = [e.reset(seed=seed) for e in envs]
    for _ in range(tiny_config.slots):
        actions = [p.act(o, i["action_mask"]) for p, (o, i) in zip(policies, states)]
        np.testing.assert_array_equal(actions[0], actions[1])
        transitions = [e.step(a) for e, a in zip(envs, actions)]
        np.testing.assert_array_equal(transitions[0][0], transitions[1][0])
        assert transitions[0][1:4] == transitions[1][1:4]
        np.testing.assert_array_equal(envs[0].node_free_at, envs[1].node_free_at)
        states = [(t[0], t[4]) for t in transitions]
    assert envs[0].records == envs[1].records
    assert envs[0].batch_records == envs[1].batch_records
    assert envs[0].summary()["energy_window"] is None
    assert envs[1].summary()["energy_window"]["total_j"] > 0


def test_invalid_trace_is_rejected_instead_of_repaired():
    with pytest.raises(AssertionError):
        integrate_power_samples([1, 0], [10, 20])

