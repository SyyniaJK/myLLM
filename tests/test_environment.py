from dataclasses import replace
import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env
from edgeinfer.allocation import EqualBandwidth
from edgeinfer.config import Device
from edgeinfer.environment import EdgeEnv
from edgeinfer.experiment import RandomPolicy
from edgeinfer.workload import ClosedLoopWorkload, Request


def request(user=0, *, arrival=0.4, slo=10, output=2):
    return Request(user, 0, arrival, 2, output, output, slo)


def test_gymnasium_api_and_determinism(tiny_config):
    check_env(EdgeEnv(tiny_config), skip_render_check=True)


def test_empty_slot_zero_reward_and_no_batches(injected_env):
    env = injected_env([])
    _, reward, _, _, info = env.step(np.zeros(2, dtype=int))
    assert reward == 0 and info["requests"] == 0
    assert env.batch_records == []
    assert env.summary()["goodput"] == 0


def test_single_user_complete_e2e(injected_env):
    env = injected_env([request()])
    env.step(np.array([1, 0]))
    row = env.records[0]
    assert row["good"] and row["delivered"]
    assert row["e2e_s"] == pytest.approx(sum(row[k] for k in
        ("ap_wait_s", "upload_s", "batch_wait_s", "gpu_ttft_s", "gpu_decode_s", "downlink_s")))
    assert row["ap_wait_s"] == 0.6


def test_oom_records_failure_and_charges_only_upload(tiny_config, injected_env):
    config = replace(tiny_config, device=Device("too_small", 1e-9, 1.0))
    env = injected_env([request()], config=config)
    _, reward, _, _, _ = env.step(np.array([1, 0]))
    assert reward == -0.5
    assert env.records[0]["status"] == "oom"
    assert env.summary()["throughput"] == 0
    assert env.ledger.summary(0, 10)["gpu_prefill_increment_j"] == 0
    assert env.ledger.summary(0, 10)["uplink_j"] > 0


def test_mask_only_request_and_idle_not_joint_memory(injected_env):
    env = injected_env([request(0), request(1)])
    env.node_free_at[0] = 5.0
    mask = env.action_masks().reshape(2, 3)
    np.testing.assert_array_equal(mask, [[True, False, True], [True, False, True]])


def test_missing_request_mask_only_reject(injected_env):
    env = injected_env([request(0)])
    np.testing.assert_array_equal(env.action_masks().reshape(2, 3)[1], [True, False, False])


def test_busy_assignment_fails_explicitly(injected_env):
    env = injected_env([request()])
    env.node_free_at[0] = 5.0
    env.step(np.array([1, 0]))
    assert env.records[0]["status"] == "invalid"
    assert env.node_free_at[0] == 5.0


def test_rejection_starts_new_thinking_at_decision(injected_env):
    env = injected_env([request()])
    env.step(np.array([0, 0]))
    assert env.records[0]["response_s"] == 1.0
    assert env.workload.next_arrival[0] > 1 or 0 in env.current_requests


def test_same_batch_same_gpu_time(injected_env):
    env = injected_env([request(0, output=2), request(1, output=4)])
    env.step(np.array([1, 1]))
    a, b = env.records
    assert a["gpu_ttft_s"] == b["gpu_ttft_s"]
    assert a["gpu_decode_s"] == b["gpu_decode_s"]


def test_strict_slo_drop_and_no_infinite_busy_state(injected_env):
    env = injected_env([request(slo=0.01)])
    env.step(np.array([1, 0]))
    row = env.records[0]
    assert row["status"] == "downlink_drop"
    assert not row["good"] and not row["delivered"] and row["inferred"]
    assert np.isfinite(env.node_free_at).all()


def test_equal_downlink_can_return_late_results(injected_env):
    env = injected_env([request(slo=0.01)], allocator=EqualBandwidth())
    env.step(np.array([1, 0]))
    assert env.records[0]["status"] == "slo_miss"
    assert env.summary()["throughput"] == 1
    assert env.summary()["goodput"] == 0


def test_node_not_released_early_and_memory_freed(tiny_config, injected_env):
    c = replace(tiny_config, device=Device("slow", 0.01, 1e-10))
    env = injected_env([request(slo=100)], config=c)
    env.step(np.array([1, 0]))
    batch = env.batch_records[0]
    assert batch["inference_end_s"] > 3
    assert not env.action_masks().reshape(2, 3)[:, 1].any()
    assert env.memory_bytes(2)[0] > 2 * c.model.parameters
    assert env.memory_bytes(batch["inference_end_s"])[0] == 2 * c.model.parameters
    assert env.node_free_at[0] >= batch["inference_end_s"]


def test_horizon_pending_and_energy_tail(tiny_config, injected_env):
    c = replace(tiny_config, slots=1, device=Device("slow", 0.01, 1e-10))
    env = injected_env([request(slo=100)], config=c)
    env.step(np.array([1, 0]))
    metrics = env.summary()
    assert metrics["goodput"] == 1
    assert metrics["goodput_within_window"] == 0
    assert metrics["pending_at_horizon"] == 1
    assert metrics["energy_tail"]["total_j"] > 0


def test_no_repeat_arrival_while_previous_request_outstanding(tiny_config):
    workload = ClosedLoopWorkload(tiny_config, 0)
    workload.next_arrival[:] = [0.4, np.inf]
    assert list(workload.arrivals(0, 1)) == [0]
    assert workload.arrivals(1, 2) == {}
    assert workload.arrivals(2, 3) == {}
    workload.finish(0, 3.5)
    assert workload.next_arrival[0] >= 3.5


def test_arrival_at_boundary_is_consumed_once(tiny_config):
    workload = ClosedLoopWorkload(tiny_config, 0)
    workload.next_arrival[:] = [1.0, np.inf]
    assert workload.arrivals(0, 1) == {}
    assert list(workload.arrivals(1, 2)) == [0]
    assert workload.arrivals(2, 3) == {}


@pytest.mark.parametrize("users,nodes", [(1, 1), (10, 2), (20, 6)])
def test_dimensions_resize_everywhere(tiny_config, users, nodes):
    env = EdgeEnv(tiny_config.resized(users=users, nodes=nodes))
    obs, info = env.reset(seed=1)
    assert len(obs) == 3 * users + 2 * users * nodes + nodes
    assert info["action_mask"].shape == (users, nodes + 1)
    env.step(np.zeros(users, dtype=int))


def test_output_realization_not_leaked_to_policy(injected_env):
    a = Request(0, 0, 0.4, 2, 4, 2, 10)
    b = replace(a, output_tokens=4)
    env1, env2 = injected_env([a]), injected_env([b])
    np.testing.assert_array_equal(env1.observation(), env2.observation())
    np.testing.assert_array_equal(env1.action_masks(), env2.action_masks())


def test_action_contract_does_not_silently_coerce(injected_env):
    env = injected_env([request()])
    with pytest.raises(AssertionError):
        env.step(np.array([1.5, 0.0]))


def test_seed_replays_full_closed_loop(tiny_config):
    trajectories = []
    for _ in range(2):
        env = EdgeEnv(tiny_config)
        obs, info = env.reset(seed=25)
        policy = RandomPolicy(7)
        for _ in range(tiny_config.slots):
            obs, _, _, _, info = env.step(policy.act(obs, info["action_mask"]))
        trajectories.append((env.records, env.batch_records, env.summary()))
    assert trajectories[0] == trajectories[1]

