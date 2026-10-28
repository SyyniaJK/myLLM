"""Frozen-input multi-method benchmark. Algorithms share one physical evaluator."""

from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
import importlib
import json
import os
from pathlib import Path
import time
import numpy as np
from .config import config_from_dict
from .allocation import TPBandwidth
from .environment import EdgeEnv
from .experiment import runtime_versions, train, write_csv, write_json
from .policies import FACTORIES
from .ppo import MaskedPPO
from .scenarios import Scenario
from .workload import Request


LEARNED = ("tp_mppo", "tp_ppo")
DEFAULT_METHODS = (*LEARNED, "random_masked", "strongest_channel", "balanced", "cap_greedy")


def _train_ablation(job):
    case, config, seed, steps, eval_episodes, output = job
    train(config, seed, steps, Path(output) / case / f"seed_{seed}",
          eval_episodes=eval_episodes, algorithm="tp_ppo")
    return {"case": case, "training_seed": seed, "steps": steps}


def train_ablations(reference_suite, output, workers):
    reference_suite, output = Path(reference_suite), Path(output)
    manifest = json.loads((reference_suite / "suite.json").read_text(encoding="utf-8"))
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "training_plan.json", {"reference_suite": str(reference_suite.resolve()),
               "algorithm": "tp_ppo", "mask_enabled": False, "steps_per_run": manifest["steps_per_run"],
               "training_seeds": manifest["training_seeds"], "cases": manifest["cases"]})
    jobs = [(name, config_from_dict(data), seed, manifest["steps_per_run"],
             len(manifest["validation_seeds"]), str(output))
            for name, data in manifest["cases"].items() for seed in manifest["training_seeds"]]
    with ProcessPoolExecutor(max_workers=workers) as executor:
        completed = list(executor.map(_train_ablation, jobs))
    write_json(output / "completed.json", completed)


def freeze_scenarios(reference_suite, seeds, output):
    source = json.loads((Path(reference_suite) / "suite.json").read_text(encoding="utf-8"))
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    seeds = [int(s) for s in seeds]
    assert seeds and len(seeds) == len(set(seeds))
    for name, data in source["cases"].items():
        config = config_from_dict(data)
        for seed in seeds:
            Scenario.generate(config, seed).save(output / name / f"scenario_{seed}")
    write_json(output / "manifest.json", {"format": 1, "scenario_seeds": seeds,
                                         "cases": source["cases"], "source_training_seeds": source["training_seeds"]})


def import_factory(specification):
    module_name, function_name = specification.split(":")
    return getattr(importlib.import_module(module_name), function_name)


def checkpoint_path(root, case, replicate):
    return Path(root) / case / f"seed_{replicate}" / "policy.pt"


def metric_row(env, decision_ms):
    summary = env.summary()
    horizon = env.config.horizon_s
    delivered = [r for r in env.records if r["delivered"] and r["response_s"] <= horizon]
    good = summary["goodput_within_window"]
    offered = summary["requests"]
    delay = np.array([r["e2e_s"] for r in delivered])
    energy = summary["energy_window"]
    return {
        "requests": offered, "goodput_window": good, "throughput_window": len(delivered),
        "goodput_ratio_window": good / offered if offered else 0.0,
        "goodput_cohort": summary["goodput"], "goodput_ratio_cohort": summary["goodput_ratio"],
        "reward": summary["reward"], "oom_requests": summary["oom_requests"],
        "rejected": summary["rejected"], "invalid": summary["invalid"],
        "downlink_dropped": summary["downlink_dropped"], "pending": summary["pending_at_horizon"],
        "energy_j": energy["total_j"], "uplink_j": energy["uplink_j"], "downlink_j": energy["downlink_j"],
        "gpu_idle_baseline_j": energy["gpu_idle_baseline_j"],
        "gpu_prefill_increment_j": energy["gpu_prefill_increment_j"],
        "gpu_decode_increment_j": energy["gpu_decode_increment_j"],
        "energy_tail_j": summary["energy_tail"]["total_j"],
        "e2e_mean_s": float(delay.mean()) if delay.size else None,
        "e2e_p95_s": float(np.quantile(delay, 0.95)) if delay.size else None,
        "slo_attainment_delivered": good / len(delivered) if delivered else None,
        "policy_decision_ms_mean": float(np.mean(decision_ms)),
        "policy_decision_ms_p95": float(np.quantile(decision_ms, 0.95)),
    }


def evaluate_episode(scenario, mode, policy, *, energy=True, allocator=None):
    env = EdgeEnv(scenario.config, energy=energy, allocator=allocator)
    obs, info = env.reset(seed=scenario.seed, options={"scenario": scenario, "workload_mode": mode})
    actions, rewards, timing = [], [], []
    for _ in range(scenario.config.slots):
        start = time.perf_counter_ns()
        action = policy.act(obs, info["action_mask"], deterministic=True)
        timing.append((time.perf_counter_ns() - start) / 1e6)
        obs, reward, _, _, info = env.step(action)
        actions.append(action.copy())
        rewards.append(reward)
    if mode == "fixed":
        expected = {(r.user, r.sequence): asdict(r) for r in scenario.offers}
        actual = {(r["user"], r["sequence"]): {k: r[k] for k in Request.__dataclass_fields__}
                  for r in env.records}
        assert len(actual) == len(env.records)
        assert actual == expected, "Every method must consume the exact saved external offers."
    return env, np.asarray(actions), np.asarray(rewards), timing


def scientific_metrics(row):
    """Wall-clock measurements are deliberately excluded from exact replay."""
    return {k: v for k, v in row.items() if not k.startswith("policy_decision_ms")}


def _make_policy(method, config, replicate, scenario_seed, case, reference_root, ablation_root, custom):
    if method in LEARNED:
        root = reference_root if method == "tp_mppo" else ablation_root
        policy = MaskedPPO.load(checkpoint_path(root, case, replicate))
        assert policy.config == config
        assert policy.use_mask == (method == "tp_mppo")
        metadata = json.loads((checkpoint_path(root, case, replicate).parent / "run.json").read_text(encoding="utf-8"))
        return policy, metadata["training_steps"]
    if method in custom:
        policy, steps = import_factory(custom[method])(config, replicate)
        assert isinstance(steps, int) and steps >= 0
        return policy, steps
    factory = FACTORIES[method]
    seed = int(np.random.default_rng(np.random.SeedSequence([replicate, scenario_seed, 60])).integers(0, 2**31))
    return factory(config, seed), 0


def _evaluate_job(job):
    case, mode, replicate, seeds, methods, scenario_root, reference_root, ablation_root, output, custom, allocators = job
    output = Path(output)
    rows = []
    for method in methods:
        for scenario_seed in seeds:
            scenario = Scenario.load(Path(scenario_root) / case / f"scenario_{scenario_seed}")
            policy, steps = _make_policy(method, scenario.config, replicate, scenario_seed, case,
                                         reference_root, ablation_root, custom)
            allocator = import_factory(allocators[method])(scenario.config) if method in allocators else TPBandwidth()
            env, actions, rewards, timing = evaluate_episode(scenario, mode, policy, allocator=allocator)
            row = {"case": case, "mode": mode, "method": method, "replicate": replicate,
                   "scenario_seed": scenario_seed, "training_steps": steps,
                   "horizon_s": scenario.config.horizon_s, **metric_row(env, timing)}
            rows.append(row)
            path = output / "episodes" / case / mode / method / f"replicate_{replicate}" / f"scenario_{scenario_seed}"
            path.mkdir(parents=True, exist_ok=True)
            write_json(path / "metrics.json", row)
            np.savez_compressed(path / "trajectory.npz", actions=actions, rewards=rewards)
            if scenario_seed == seeds[0]:
                # One complete trace per method/replicate/case/mode; any other run can be replayed.
                if env.records:
                    write_csv(path / "requests.csv", env.records)
                write_json(path / "batches.json", env.batch_records)
                write_json(path / "power_intervals.json", env.ledger.rows())
                replay_policy, _ = _make_policy(method, scenario.config, replicate, scenario_seed, case,
                                                reference_root, ablation_root, custom)
                replay_allocator = import_factory(allocators[method])(scenario.config) if method in allocators else TPBandwidth()
                replay_env, aa, rr, tt = evaluate_episode(scenario, mode, replay_policy, allocator=replay_allocator)
                np.testing.assert_array_equal(actions, aa)
                np.testing.assert_array_equal(rewards, rr)
                assert env.records == replay_env.records and env.batch_records == replay_env.batch_records
                assert scientific_metrics(metric_row(env, timing)) == scientific_metrics(metric_row(replay_env, tt))
                write_json(path / "replay_check.json", {"exact": True, "actions": True, "rewards": True,
                           "requests": True, "batches": True, "scientific_metrics": True,
                           "wall_clock_excluded": True})
        print(f"Benchmark {case}/{mode}/replicate_{replicate}: {method} completed", flush=True)
    return rows


def run_benchmark(scenario_root, reference_root, ablation_root, output, *, methods=DEFAULT_METHODS,
                  modes=("closed_loop", "fixed"), replicates=(1, 2, 3), workers=3,
                  bootstrap_samples=2000, stats_seed=171, custom=None, allocators=None):
    scenario_root, reference_root, ablation_root = map(Path, (scenario_root, reference_root, ablation_root))
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    custom = {} if custom is None else dict(custom)
    allocators = {} if allocators is None else dict(allocators)
    assert not set(custom) & set(DEFAULT_METHODS), "Custom method names must be distinct from built-ins."
    assert set(allocators) <= set(custom), "Allocator overrides belong to explicitly named custom methods."
    assert methods and len(methods) == len(set(methods)) and set(methods) <= set(DEFAULT_METHODS) | set(custom)
    assert len(replicates) == len(set(replicates)) and replicates
    assert modes and set(modes) <= {"closed_loop", "fixed"} and len(modes) == len(set(modes))
    scenarios = json.loads((scenario_root / "manifest.json").read_text(encoding="utf-8"))
    manifest = {"format": 1, "methods": list(methods), "modes": list(modes), "replicates": list(replicates),
                "cases": scenarios["cases"], "scenario_seeds": scenarios["scenario_seeds"], "custom": custom,
                "allocators": allocators,
                "scenario_root": Path(os.path.relpath(scenario_root, output)).as_posix(),
                "reference_root": Path(os.path.relpath(reference_root, output)).as_posix(),
                "ablation_root": Path(os.path.relpath(ablation_root, output)).as_posix(),
                "bootstrap_samples": bootstrap_samples, "stats_seed": stats_seed,
                "runtime": runtime_versions(), "primary_boundary": "fixed_observation_window",
                "learned_training_workload": "closed_loop", "fixed_mode_role": "frozen_policy_open_loop_transfer"}
    # One budget per learned method/replicate/case; no method receives a larger training allocation.
    for case in manifest["cases"]:
        for replicate in replicates:
            budgets = []
            for method in set(methods) & set(LEARNED):
                root = reference_root if method == "tp_mppo" else ablation_root
                run = json.loads((checkpoint_path(root, case, replicate).parent / "run.json").read_text(encoding="utf-8"))
                assert config_from_dict(run["config"]) == config_from_dict(manifest["cases"][case])
                budgets.append(run["training_steps"])
            for method in set(methods) & set(custom):
                _, steps = import_factory(custom[method])(config_from_dict(manifest["cases"][case]), replicate)
                assert isinstance(steps, int) and steps >= 0
                if steps:
                    budgets.append(steps)
            assert len(set(budgets)) <= 1, "Learned methods must have identical interaction budgets."
    write_json(output / "benchmark.json", manifest)
    jobs = [(case, mode, replicate, manifest["scenario_seeds"], list(methods), str(scenario_root),
             str(reference_root), str(ablation_root), str(output), custom, allocators)
            for case in manifest["cases"] for mode in modes for replicate in replicates]
    rows = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        for completed in executor.map(_evaluate_job, jobs):
            rows.extend(completed)
            write_csv(output / "episodes.csv", rows)
    write_json(output / "episodes.json", rows)
    from .benchmark_stats import summarize_benchmark
    summarize_benchmark(output)
    from .benchmark_plotting import plot_benchmark
    plot_benchmark(output)
    return rows


def replay_benchmark(directory, case, mode, method, replicate, scenario_seed, output):
    directory, output = Path(directory), Path(output)
    plan = json.loads((directory / "benchmark.json").read_text(encoding="utf-8"))
    scenario = Scenario.load(directory / plan["scenario_root"] / case / f"scenario_{scenario_seed}")
    policy, _ = _make_policy(method, scenario.config, replicate, scenario_seed, case,
                            directory / plan["reference_root"], directory / plan["ablation_root"], plan["custom"])
    allocator = import_factory(plan["allocators"][method])(scenario.config) if method in plan["allocators"] else TPBandwidth()
    env, actions, rewards, timing = evaluate_episode(scenario, mode, policy, allocator=allocator)
    original = directory / "episodes" / case / mode / method / f"replicate_{replicate}" / f"scenario_{scenario_seed}"
    saved = json.loads((original / "metrics.json").read_text(encoding="utf-8"))
    with np.load(original / "trajectory.npz", allow_pickle=False) as trace:
        np.testing.assert_array_equal(actions, trace["actions"])
        np.testing.assert_array_equal(rewards, trace["rewards"])
    expected = {k: saved[k] for k in metric_row(env, timing)}
    assert scientific_metrics(metric_row(env, timing)) == scientific_metrics(expected)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "replay.json", {"exact": True, "source": str(original.resolve()), **metric_row(env, timing)})
    if env.records:
        write_csv(output / "requests.csv", env.records)
    write_json(output / "batches.json", env.batch_records)
    return env
