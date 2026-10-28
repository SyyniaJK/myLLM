"""G5 fixed-input diagnostics. No new learning or changes to baseline physics."""

from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, replace
from math import fsum
from pathlib import Path
import json
import subprocess
import numpy as np
import torch
from .allocation import TPBandwidth
from .benchmark import DEFAULT_METHODS, evaluate_episode, metric_row, scientific_metrics
from .config import load_config
from .environment import EdgeEnv
from .experiment import runtime_versions, write_csv, write_json
from .physics import batch_cost, full_band_rates
from .policies import FACTORIES, CapGreedyPolicy
from .ppo import MaskedPPO
from .scenarios import Scenario
from .workload import Request


class EqualUplink(TPBandwidth):
    """Only uplink changes; downlink still uses the common TP P3 rule."""

    def uplink(self, bits, full_rates):
        return np.full(len(bits), 1 / len(bits))


ALLOCATORS = {"tp": TPBandwidth, "equal_uplink": EqualUplink}
ESTIMATES = {"cap": (False, 1.0), "conditional_mean": (False, 0.75),
             "half_cap": (False, 0.5), "oracle": (True, 1.0),
             "oracle_half": (True, 0.5), "oracle_over": (True, 1.25)}


def make_policy(method, config, replicate, seed):
    if method in ("tp_mppo", "tp_ppo"):
        source = "g123_final" if method == "tp_mppo" else "g4_training"
        case = {15: "default", 10: "users_10"}[config.users]
        agent = MaskedPPO.load(Path("results") / source / case / f"seed_{replicate}" / "policy.pt")
        assert (agent.config.users, agent.config.nodes) == (config.users, config.nodes)
        assert agent.use_mask == (method == "tp_mppo")
        return agent
    random_seed = int(np.random.default_rng(np.random.SeedSequence([replicate, seed, 60])).integers(0, 2**31))
    return FACTORIES[method](config, random_seed)


def attenuate(scenario, db):
    factor = 10 ** (-db / 10)
    return replace(scenario, up=scenario.up * factor, down=scenario.down * factor)


def diagnostic_metrics(env):
    # Wall-clock fields are not part of G5; use the same G4 physical metrics.
    row = scientific_metrics(metric_row(env, [0.0]))
    c, p = env.config, env.config.energy
    intervals = env.ledger.intervals
    gpu = [i for i in intervals if i.component.startswith("gpu_")]
    radio = [i for i in intervals if i.component in ("uplink_j", "downlink_j")]
    full = lambda items: fsum((i.end_s - i.start_s) * i.power_w for i in items)
    batches = [b for b in env.batch_records if not b["oom"]]
    by_user_slot = {(r["slot"], r["user"]): r for r in env.records}
    solo = []
    for b in batches:
        for user in b["users"]:
            r = by_user_slot[b["slot"], user]
            cost = batch_cost(c.model, [r["input_tokens"]], [r["output_tokens"]], c.flops_s)
            solo.append(c.gpus_per_node * ((p.prefill_w_per_gpu - p.idle_w_per_gpu) * cost.prefill_s
                        + (p.decode_w_per_gpu - p.idle_w_per_gpu) * cost.decode_s))
    rf = row["uplink_j"] + row["downlink_j"]
    gpu_charged = fsum(c.gpus_per_node * ((p.prefill_w_per_gpu - p.idle_w_per_gpu) * b["prefill_s"]
                      + (p.decode_w_per_gpu - p.idle_w_per_gpu) * b["decode_s"]) for b in batches)
    row.update({
        "fixed_j": row["gpu_idle_baseline_j"], "radio_window_j": rf,
        "variable_window_j": row["energy_j"] - row["gpu_idle_baseline_j"],
        "without_rf_window_j": row["energy_j"] - rf,
        "gpu_increment_window_j": row["gpu_prefill_increment_j"] + row["gpu_decode_increment_j"],
        "gpu_increment_full_j": gpu_charged, "radio_full_j": full(radio),
        "gpu_increment_full_ledger_j": full(gpu),
        "variable_full_j": gpu_charged + full(radio), "gpu_solo_full_j": fsum(solo),
        "gpu_increment_tail_j": fsum(i.energy(c.horizon_s, i.end_s) for i in gpu if i.end_s > c.horizon_s),
        "gpu_utilization": fsum(max(0.0, min(c.horizon_s, b["inference_end_s"])
                                       - b["inference_start_s"]) for b in batches) / (c.nodes * c.horizon_s),
        "batches": len(env.batch_records), "inferred_batches": len(batches),
        "batch_upload_sum_s": fsum(b["upload_end_s"] - b["upload_start_s"] for b in env.batch_records),
        "batch_wait_sum_s": fsum(r["batch_wait_s"] for r in env.records if r["batch_wait_s"] is not None),
    })
    return row


def scan_case(think, db):
    return f"think_{think:g}_atten_{db:g}"


def save_trace(output, key, actions, rewards):
    path = Path(output) / "trajectories" / f"{key}.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, actions=actions, rewards=rewards)


def _scan_job(job):
    output, think, db, allocator_name, replicate, seeds = job
    rows = []
    for method in DEFAULT_METHODS:
        for seed in seeds:
            base = Scenario.load(Path(output) / "inputs" / f"think_{think:g}" / f"scenario_{seed}")
            scenario = attenuate(base, db)
            policy = make_policy(method, scenario.config, replicate, seed)
            env, actions, rewards, _ = evaluate_episode(scenario, "fixed", policy,
                                                       allocator=ALLOCATORS[allocator_name]())
            case = scan_case(think, db)
            key = f"{case}__{allocator_name}__{method}__r{replicate}__s{seed}"
            row = {"experiment": "g5_3", "case": case, "method": method, "allocator": allocator_name,
                   "replicate": replicate, "scenario_seed": seed, "mean_think_s": think,
                   "attenuation_db": db, "horizon_s": scenario.config.horizon_s,
                   "offered_per_s": len(scenario.offers) / scenario.config.horizon_s,
                   "trace_key": key, **diagnostic_metrics(env)}
            rows.append(row)
            save_trace(output, key, actions, rewards)
    print(f"G5-3 {scan_case(think, db)}/{allocator_name}/seed {replicate}: {len(rows)} episodes", flush=True)
    return rows


def snapshot_scenario(seed, slo, db):
    config = replace(load_config("configs/paper_aligned.toml"), users=10, slots=12)
    offers = tuple(Request(u, 0, 0.5, 128 if u < 3 else 512,
                           128 if u < 3 else 1024, 128 if u < 3 else 1024, slo) for u in range(6))
    return attenuate(replace(Scenario.generate(config, seed), offers=offers), db)


def execute_snapshot(scenario, action, allocator=None):
    env = EdgeEnv(scenario.config, allocator=allocator)
    env.reset(seed=scenario.seed, options={"scenario": scenario, "workload_mode": "fixed"})
    env.step(np.asarray(action, dtype=int))
    return env


def composition_controls(plan, output):
    rows = []
    placements = {"grouped": [1, 1, 1, 2, 2, 2], "mixed": [1, 1, 2, 1, 2, 2]}
    for seed in plan["seeds"]:
        for slo in plan["snapshot_slo_s"]:
            for db in plan["snapshot_attenuation_db"]:
                scenario = snapshot_scenario(seed, slo, db)
                for name, choices in placements.items():
                    for allocator_name in plan["allocators"]:
                        action = np.array(choices + [0] * 4)
                        env = execute_snapshot(scenario, action, ALLOCATORS[allocator_name]())
                        rows.append({"experiment": "g5_4", "case": f"slo_{slo:g}_atten_{db}",
                            "method": name, "allocator": allocator_name, "replicate": 1,
                            "scenario_seed": seed, "slo_s": slo, "attenuation_db": db,
                            **diagnostic_metrics(env)})
    write_csv(Path(output) / "composition.csv", rows)
    return rows


class LengthEstimatePolicy:
    """Ablate the memory-greedy proxy; oracle modes explicitly access future truth."""

    def __init__(self, scenario, mode):
        self.config = scenario.config
        self.policy = CapGreedyPolicy(scenario.config, 0)
        oracle, self.factor = ESTIMATES[mode]
        self.truth = ({(int(r.arrival_s // scenario.config.slot_s), r.user): r.output_tokens
                       for r in scenario.offers} if oracle else None)
        self.slot = 0

    def act(self, observation, mask, *, deterministic=True):
        obs, n = observation.copy(), self.config.users
        active = np.flatnonzero(obs[:n] > 0)
        base = obs[n:2*n].copy()
        if self.truth is not None:
            base[active] = [self.truth[self.slot, int(u)] for u in active]
        obs[n:2*n][active] = np.maximum(1, np.rint(self.factor * base[active]))
        self.slot += 1
        return self.policy.act(obs, mask, deterministic=deterministic)


def length_controls(plan, output):
    rows = []
    for seed in plan["seeds"]:
        base = Scenario.load(Path(output) / "inputs" / "think_0.5" / f"scenario_{seed}")
        for memory in plan["memory_gb"]:
            scenario = replace(base, config=replace(base.config, device=replace(base.config.device, memory_gb=memory)))
            for mode in plan["estimators"]:
                policy = LengthEstimatePolicy(scenario, mode)
                env, actions, rewards, _ = evaluate_episode(scenario, "fixed", policy)
                oracle, factor = ESTIMATES[mode]
                truth = np.array([r.output_tokens for r in scenario.offers])
                basis = truth if oracle else np.array([r.output_cap for r in scenario.offers])
                predicted = np.maximum(1, np.rint(factor * basis))
                key = f"length__mem{memory:g}__{mode}__s{seed}"
                save_trace(output, key, actions, rewards)
                rows.append({"experiment": "g5_5", "case": f"memory_{memory:g}", "method": mode,
                    "allocator": "tp", "replicate": 1, "scenario_seed": seed,
                    "memory_gb": memory, "future_information": oracle,
                    "length_mae_tokens": float(np.abs(predicted - truth).mean()),
                    "underestimated_requests": int(np.sum(predicted < truth)), "trace_key": key,
                    **diagnostic_metrics(env)})
    write_csv(Path(output) / "length.csv", rows)
    return rows


def energy_audit(output):
    rows = json.loads(Path("results/g4/benchmark/episodes.json").read_text(encoding="utf-8"))
    decomposition = []
    for r in rows:
        fixed = r["gpu_idle_baseline_j"]
        rf = r["uplink_j"] + r["downlink_j"]
        decomposition.append({k: r[k] for k in ("case", "mode", "method", "replicate", "scenario_seed",
            "goodput_window", "energy_j")} | {"fixed_j": fixed, "radio_j": rf,
            "gpu_increment_j": r["gpu_prefill_increment_j"] + r["gpu_decode_increment_j"],
            "fixed_fraction": fixed / r["energy_j"], "radio_fraction": rf / r["energy_j"],
            "variable_fraction_upper_bound": 1 - fixed / r["energy_j"]})
    write_csv(Path(output) / "energy_decomposition.csv", decomposition)
    config = load_config("configs/paper_aligned.toml")
    audit = {"calibration_status": "not_hardware_calibrated", "cuda_available": torch.cuda.is_available(),
        "phase_power_w": asdict(config.energy), "bits_per_token": config.bits_per_token,
        "effective_tflops": config.device.tflops,
        "expected_default_fixed_j": config.nodes * config.gpus_per_node * config.energy.idle_w_per_gpu * config.horizon_s,
        "g4_rows_checked": len(rows), "fixed_baseline_per_case": True,
        "radio_noise_model": "total_noise_power_and_linear_bandwidth_fraction_paper_approximation",
        "missing_calibration": ["phase power traces", "effective GPU throughput and memory bandwidth",
                                "wire payload and protocol overhead", "channel/link measurements"],
        "no_sleep_or_dvfs_control": True,
        "same_batch_full_gpu_energy_depends_on_lengths_and_group_not_upload_start": True}
    configs = json.loads(Path("results/g4/benchmark/benchmark.json").read_text(encoding="utf-8"))["cases"]
    for r in rows:
        c = configs[r["case"]]
        assert r["gpu_idle_baseline_j"] == c["nodes"] * c["gpus_per_node"] * c["energy"]["idle_w_per_gpu"] * c["slots"] * c["slot_s"]
    write_json(Path(output) / "model_audit.json", audit)


def run_g5(config_path, output, workers=3):
    output = Path(output)
    assert not output.exists(), "Use a fresh directory; historical experiments are not overwritten."
    output.mkdir(parents=True)
    plan = json.loads(Path(config_path).read_text(encoding="utf-8"))
    assert plan["mean_think_s"] == [8.0, 2.0, 0.5]
    plan.update({"format": 1, "methods": list(DEFAULT_METHODS), "base_config": asdict(load_config("configs/paper_aligned.toml")),
        "runtime": runtime_versions(), "code_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "workload_mode": "fixed", "learned_policy_status": "frozen_nominal_checkpoint_transfer",
        "reference_suite": "results/g123_final", "ablation_root": "results/g4_training"})
    write_json(output / "plan.json", plan)
    energy_audit(output)
    base = load_config("configs/paper_aligned.toml")
    for think in plan["mean_think_s"]:
        for seed in plan["seeds"]:
            Scenario.generate(replace(base, mean_think_s=think), seed).save(
                output / "inputs" / f"think_{think:g}" / f"scenario_{seed}")
    jobs = [(str(output), think, db, allocator, replicate, plan["seeds"])
            for think in plan["mean_think_s"] for db in plan["attenuation_db"]
            for allocator in plan["allocators"] for replicate in plan["replicates"]]
    rows = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        for completed in executor.map(_scan_job, jobs):
            rows.extend(completed)
            write_csv(output / "scan.csv", rows)
    write_json(output / "scan.json", rows)
    print("G5-3 complete; running controlled snapshots and length errors", flush=True)
    composition_controls(plan, output)
    length_controls(plan, output)
    from .snapshot import exact_controls
    exact_controls(plan, output)
    from .gap_analysis import analyze_g5
    analyze_g5(output)
    return rows


def replay_g5(directory, case, method, allocator, replicate, seed, output):
    directory, output = Path(directory), Path(output)
    rows = json.loads((directory / "scan.json").read_text(encoding="utf-8"))
    saved, = [r for r in rows if (r["case"], r["method"], r["allocator"], r["replicate"], r["scenario_seed"])
              == (case, method, allocator, replicate, seed)]
    base = Scenario.load(directory / "inputs" / f"think_{saved['mean_think_s']:g}" / f"scenario_{seed}")
    scenario = attenuate(base, saved["attenuation_db"])
    env, actions, rewards, _ = evaluate_episode(scenario, "fixed", make_policy(method, scenario.config, replicate, seed),
                                               allocator=ALLOCATORS[allocator]())
    with np.load(directory / "trajectories" / f"{saved['trace_key']}.npz", allow_pickle=False) as trace:
        np.testing.assert_array_equal(actions, trace["actions"])
        np.testing.assert_array_equal(rewards, trace["rewards"])
    row = diagnostic_metrics(env)
    assert all(saved[k] == value for k, value in row.items())
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "replay.json", {"exact": True, "case": case, "method": method,
                                        "allocator": allocator, "replicate": replicate, "seed": seed, **row})
    return row
