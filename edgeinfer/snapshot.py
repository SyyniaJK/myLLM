"""Exhaustive single-decision placement frontier under the common P2/P3 rule."""

from dataclasses import replace
from itertools import product
from math import fsum
from pathlib import Path
import numpy as np
from .benchmark import DEFAULT_METHODS
from .diagnostics import diagnostic_metrics, execute_snapshot, make_policy, snapshot_scenario
from .environment import EdgeEnv
from .experiment import write_csv


def subset_costs(scenario, *, use_caps=False):
    requests = tuple(sorted(scenario.offers, key=lambda r: r.user))
    assert len({r.user for r in requests}) == len(requests)
    assert all(r.arrival_s < scenario.config.slot_s for r in requests)
    tables = []
    for node in range(scenario.config.nodes):
        table = {0: (0, 0.0)}
        for subset in range(1, 1 << len(requests)):
            group = [r for i, r in enumerate(requests) if subset & (1 << i)]
            if use_caps:
                group = [replace(r, output_tokens=r.output_cap) for r in group]
            env = EdgeEnv(scenario.config)
            env.reset(seed=scenario.seed, options={"scenario": scenario, "workload_mode": "fixed"})
            env._execute_batch(node, group)
            table[subset] = (sum(r["good"] for r in env.records), diagnostic_metrics(env)["variable_full_j"])
        tables.append(table)
    return requests, tables


def plan_cost(choices, tables):
    subsets = [0] * len(tables)
    for i, choice in enumerate(choices):
        if choice:
            subsets[choice - 1] |= 1 << i
    costs = [table[subset] for table, subset in zip(tables, subsets)]
    return sum(x[0] for x in costs), fsum(x[1] for x in costs)


def exact_frontier(tables, request_count):
    frontier = {}
    best = None
    for choices in product(range(len(tables) + 1), repeat=request_count):
        good, energy = plan_cost(choices, tables)
        candidate = (energy, choices)
        if good not in frontier or candidate < frontier[good]:
            frontier[good] = candidate
        key = (-good, energy, choices)
        if best is None or key < best:
            best = key
    return best[2], frontier


def exact_controls(plan, output):
    rows, frontiers = [], []
    for seed in plan["seeds"]:
        for slo in plan["snapshot_slo_s"]:
            for db in plan["snapshot_attenuation_db"]:
                scenario = snapshot_scenario(seed, slo, db)
                requests, actual = subset_costs(scenario)
                _, conservative = subset_costs(scenario, use_caps=True)
                oracle, frontier = exact_frontier(actual, len(requests))
                cap_plan, _ = exact_frontier(conservative, len(requests))
                max_good = max(frontier)
                case = f"slo_{slo:g}_atten_{db}"
                for g, (energy, choices) in sorted(frontier.items()):
                    frontiers.append({"case": case, "scenario_seed": seed, "slo_s": slo,
                        "attenuation_db": db, "goodput": g, "minimum_variable_full_j": energy,
                        "choices": str(list(choices)), "enumerated_placements": 4 ** len(requests)})
                choices_list = [(method, replicate, None) for method in DEFAULT_METHODS for replicate in plan["replicates"]]
                choices_list += [("oracle_exact", 1, oracle), ("cap_exact", 1, cap_plan)]
                for method, replicate, choices in choices_list:
                    if choices is None:
                        env = EdgeEnv(scenario.config)
                        obs, info = env.reset(seed=seed, options={"scenario": scenario, "workload_mode": "fixed"})
                        action = make_policy(method, scenario.config, replicate, seed).act(obs, info["action_mask"])
                        choices = tuple(int(action[r.user]) for r in requests)
                    action = np.zeros(scenario.config.users, dtype=int)
                    for request, choice in zip(requests, choices):
                        action[request.user] = choice
                    env = execute_snapshot(scenario, action)
                    metric = diagnostic_metrics(env)
                    good, charged = plan_cost(choices, actual)
                    assert good == metric["goodput_window"]
                    assert max_good >= good
                    rows.append({"experiment": "g5_6", "case": case, "method": method,
                        "replicate": replicate, "scenario_seed": seed, "slo_s": slo, "attenuation_db": db,
                        "future_information": method == "oracle_exact", "enumerated_placements": 4 ** len(requests),
                        "oracle_goodput": max_good, "service_gap": max_good - good,
                        "same_service_energy_gap_j": charged - frontier[good][0],
                        "charged_variable_full_j": charged, "choices": str(list(choices)), **metric})
                print(f"G5-6 seed {seed}/{case}: all {4**len(requests)} placements evaluated", flush=True)
    write_csv(Path(output) / "exact.csv", rows)
    write_csv(Path(output) / "exact_frontier.csv", frontiers)
    return rows, frontiers
