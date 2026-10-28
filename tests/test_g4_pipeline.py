from dataclasses import asdict, replace
import json
from pathlib import Path
import numpy as np
import pytest
from edgeinfer.benchmark import (DEFAULT_METHODS, _evaluate_job, freeze_scenarios,
                                 replay_benchmark, run_benchmark, scientific_metrics)
from edgeinfer.benchmark_stats import summarize_benchmark
from edgeinfer.experiment import train, write_json


@pytest.fixture
def comparison_files(tiny_config, tmp_path):
    config = replace(tiny_config, ppo=replace(tiny_config.ppo, hidden=16, rollout_steps=16, minibatch=8, epochs=2))
    reference, ablation, scenarios = (tmp_path / name for name in ("reference", "ablation", "scenarios"))
    reference.mkdir()
    plan = {"steps_per_run": 32, "training_seeds": [1, 2], "validation_seeds": [10000],
            "test_seeds": [20000], "cases": {"tiny": asdict(config)}}
    write_json(reference / "suite.json", plan)
    for seed in (1, 2):
        train(config, seed, 32, reference / "tiny" / f"seed_{seed}", eval_every=16, eval_episodes=1)
        train(config, seed, 32, ablation / "tiny" / f"seed_{seed}", eval_every=16, eval_episodes=1, algorithm="tp_ppo")
    freeze_scenarios(reference, [7, 8], scenarios)
    return reference, ablation, scenarios


def test_method_and_scenario_order_do_not_change_scientific_results(comparison_files, tmp_path):
    reference, ablation, scenarios = comparison_files
    methods = ["random_masked", "balanced", "cyclic"]
    custom = {"cyclic": "examples.custom_policy:build"}
    a = _evaluate_job(("tiny", "fixed", 1, [7, 8], methods, str(scenarios), str(reference),
                       str(ablation), str(tmp_path / "a"), custom, {}))
    b = _evaluate_job(("tiny", "fixed", 1, [8, 7], methods[::-1], str(scenarios), str(reference),
                       str(ablation), str(tmp_path / "b"), custom, {}))
    key = lambda r: (r["method"], r["scenario_seed"])
    assert {key(r): scientific_metrics(r) for r in a} == {key(r): scientific_metrics(r) for r in b}


def test_full_pipeline_registers_external_policy_and_replays(comparison_files, tmp_path):
    reference, ablation, scenarios = comparison_files
    output = tmp_path / "benchmark"
    methods = [*DEFAULT_METHODS, "cyclic"]
    rows = run_benchmark(scenarios, reference, ablation, output, methods=methods, replicates=[1, 2],
                         workers=1, bootstrap_samples=40, custom={"cyclic": "examples.custom_policy:build"},
                         allocators={"cyclic": "examples.custom_policy:equal_resources"})
    assert len(rows) == 1 * 2 * 7 * 2 * 2
    before = json.loads((output / "summary.json").read_text())
    summarize_benchmark(output)
    assert before == json.loads((output / "summary.json").read_text())
    env = replay_benchmark(output, "tiny", "fixed", "cyclic", 1, 8, tmp_path / "replay")
    assert env.summary()["requests"] >= 0
    assert json.loads((tmp_path / "replay" / "replay.json").read_text())["exact"]
    assert (output / "figures" / "performance.png").stat().st_size > 0
    assert (output / "figures" / "paired.svg").stat().st_size > 0


def test_unequal_training_budgets_fail_the_explicit_contract(comparison_files, tmp_path):
    reference, ablation, scenarios = comparison_files
    path = ablation / "tiny" / "seed_1" / "run.json"
    data = json.loads(path.read_text())
    data["training_steps"] = 33
    write_json(path, data)
    with pytest.raises(AssertionError, match="identical interaction budgets"):
        run_benchmark(scenarios, reference, ablation, tmp_path / "bad", replicates=[1, 2], workers=1)
