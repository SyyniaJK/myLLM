import argparse
import csv
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
import json
from pathlib import Path
import numpy as np
from .config import load_config
from .energy import integrate_power_samples
from .environment import EdgeEnv
from .experiment import RandomPolicy, evaluate, mean_metrics, train, write_csv, write_json
from .ppo import MaskedPPO


def verify_passive_energy(config, agent, seeds):
    episodes = []
    for seed in seeds:
        off, on = EdgeEnv(config, energy=False), EdgeEnv(config, energy=True)
        obs0, info0 = off.reset(seed=seed)
        obs1, info1 = on.reset(seed=seed)
        for _ in range(config.slots):
            np.testing.assert_array_equal(obs0, obs1)
            a0 = agent.act(obs0, info0["action_mask"])
            a1 = agent.act(obs1, info1["action_mask"])
            np.testing.assert_array_equal(a0, a1)
            obs0, r0, term0, trunc0, info0 = off.step(a0)
            obs1, r1, term1, trunc1, info1 = on.step(a1)
            assert (r0, term0, trunc0) == (r1, term1, trunc1)
            np.testing.assert_array_equal(obs0, obs1)
            np.testing.assert_array_equal(info0["action_mask"], info1["action_mask"])
            np.testing.assert_array_equal(off.node_free_at, on.node_free_at)
            np.testing.assert_array_equal(off.memory_bytes(off.decision_s), on.memory_bytes(on.decision_s))
        assert off.records == on.records and off.batch_records == on.batch_records
        nonenergy = [k for k in off.summary() if not k.startswith("energy")]
        assert {k: off.summary()[k] for k in nonenergy} == {k: on.summary()[k] for k in nonenergy}
        episodes.append({"seed": seed, "transitions": config.slots, "exactly_equal": True,
                         "energy_window": on.summary()["energy_window"]})
    return {"passed": True, "episodes": episodes,
            "comparisons": ["actions", "observations", "masks", "rewards", "node_release_times",
                            "memory", "request_outcomes", "batch_outcomes", "nonenergy_metrics"]}


def suite(args):
    base = load_config(args.config)
    cases = [("default", base)]
    cases += [(f"users_{n}", base.resized(users=n)) for n in args.users if n != base.users]
    cases += [(f"nodes_{n}", base.resized(nodes=n)) for n in args.nodes if n != base.nodes]
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "suite.json", {"steps_per_run": args.steps, "training_seeds": args.seeds,
                                      "validation_seeds": list(range(10_000, 10_000 + args.eval_episodes)),
                                      "test_seeds": list(range(20_000, 20_000 + args.eval_episodes)),
                                      "cases": {name: asdict(c) for name, c in cases}})
    jobs = [(name, config, seed, args.steps, output, args.eval_every, args.eval_episodes)
            for name, config in cases for seed in args.seeds]
    summaries = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        for row in executor.map(run_case, jobs):
            summaries.append(row)
            write_csv(output / "summary.csv", summaries)
            write_report(output, summaries, args)
    for name, config in cases:
        diagnostic = evaluate(config, RandomPolicy(0), range(20_000, 20_000 + args.eval_episodes))
        write_json(output / name / "random_masked_diagnostic.json", diagnostic)
    from .plotting import plot_suite
    plot_suite(output)
    print(f"Suite finished: {len(summaries)} trained policies. Results: {output.resolve()}", flush=True)


def run_case(job):
    name, config, seed, steps, output, eval_every, eval_episodes = job
    path = output / name / f"seed_{seed}"
    agent, curve, episodes = train(config, seed, steps, path,
                                   eval_every=eval_every, eval_episodes=eval_episodes)
    loaded = MaskedPPO.load(path / "policy.pt")
    replay = evaluate(config, loaded, range(20_000, 20_000 + eval_episodes))
    assert replay == episodes, "Checkpoint reload must reproduce the saved test results exactly."
    passive = verify_passive_energy(config, loaded, range(30_000, 30_000 + eval_episodes))
    write_json(path / "g3_invariance.json", passive)
    return {"case": name, "training_seed": seed, "users": config.users, "nodes": config.nodes,
            "training_steps": steps, "checkpoint_replay_equal": True,
            "g3_exact_invariance": passive["passed"],
            "initial_validation_reward": curve[0]["reward"],
            "final_validation_reward": curve[-1]["reward"], **mean_metrics(episodes)}


def write_report(output, summaries, args):
    table = []
    for name in dict.fromkeys(row["case"] for row in summaries):
        rows = [x for x in summaries if x["case"] == name]
        gp = np.array([x["goodput_ratio"] for x in rows]) * 100
        reward = np.array([x["reward"] for x in rows])
        sd = gp.std(ddof=1) if len(gp) > 1 else 0.0
        energy = np.mean([x["energy_window_j"] for x in rows])
        table.append(f"| {name} | {len(rows)} | {gp.mean():.3f} ± {sd:.3f} | {reward.mean():.3f} | {energy:.3f} |")
    defaults = [x for x in summaries if x["case"] == "default"]
    gains = [f"seed {x['training_seed']}: {x['initial_validation_reward']:.3f} → {x['final_validation_reward']:.3f}" for x in defaults]
    text = f"""# G1–G3 实验结果

平台：EdgeInferenceLab。此文件由当前实际实验生成；结果不使用原目录的历史数组。

## G1：系统与关键公式

人工构造测试覆盖论文计算量、FP16 权重与输入/输出 KV、上下行独立信道、请求去重、同步批次、节点释放、SLO、PPO 联合概率、随机种子和能耗积分。
测试命令与完整结果见 `results/g1_tests.txt`。公式、假设、原代码差异见 `docs/model_contract.md` 与 `docs/audit.md`。

## G2：真实训练与独立评估

每个策略训练 {args.steps:,} 个环境交互步；每个 episode 为 200 个时隙（具体配置保存在每次 run.json）。不同场景使用相同训练预算。
训练 seed 为 {args.seeds}；每个训练策略在 {args.eval_episodes} 个独立测试环境 seed 上评估。
验证环境使用 10000 起的 seed；测试使用 20000 起的 seed。两者与训练环境分离。
下表的 ± 是训练 seed 之间的样本标准差，先在每个策略的测试 episodes 内取均值；不是置信区间。

| 场景 | 已完成训练 seed 数 | Goodput 比例（%） | episode reward | 固定窗口估算能耗（J） |
|---|---:|---:|---:|---:|
{chr(10).join(table)}

默认场景验证奖励变化：{'; '.join(gains)}。
`learning_curve.csv` 保存每个检查点的原始数值；`figures/learning.png` 显示跨训练 seed 均值与标准差。
这些结果用于检查本平台的训练与规模变化，不证明论文报告的 33.3%–87.5% 优势已被复现；原论文四种对照算法不属于本次 G1–G3 实现范围。

## G3：纯能耗观察器

已完成的 {len(summaries)} 个策略均在额外的 {args.eval_episodes} 个环境 seed 上执行开/关能耗双轨迹检查。
动作、观测、掩码、reward、节点释放、显存、逐请求结果和逐批次结果完全一致；重新加载策略也精确重现已保存测试结果。
逐项记录保存在各策略目录的 `g3_invariance.json`。

通信能耗使用固定功率 × 实际传输时间。GPU 能耗为整个固定窗口的空闲基线，加上 prefill/decode 时段高于空闲功率的增量；批次 GPU 能耗只计一次。
默认 15/60/45 W 是演示性空闲/prefill/decode 相位功率，不是 NVIDIA L2 的实测数据。能耗用于验证核算流程，尚不能支持真实硬件节能幅度结论。
`energy_window` 仅覆盖 [0, horizon]，`energy_tail` 单独报告最后一批未完成工作的尾部；两者不会混算。

## 论文复现边界

1. 原文未给出的带宽、发射功率、载荷 bits/token、thinking time、FFN 维度及算力数值在配置与模型契约中公开记录。
2. 采用论文的理想 FLOPs/算力时延，没有偷偷改成 roofline、Vidur 或实测时延，也没有调节物理参数来逼近曲线。
3. 原代码在下行无法分配带宽时没有完整的用户结束语义。当前采用推理完成时失败通知、结果丢弃。`inference_completed` 与真正返回输出的 `throughput` 分开统计。该假设会影响闭环负载及图4中的吞吐差距。
4. TP-MPPO 的 P3 选中集合不变，沿用导师实现把剩余下行带宽按比例分给选中用户的处理。这是原文未指定的最优解择优规则。
5. 多 GPU 配置仅为显存和理想 FLOPs 的线性汇总，不代表已模拟张量并行通信。
6. 工作负载为闭环用户；同 seed 对齐随机规则，并不使不同策略产生完全相同的请求到达时刻。
"""
    (output / "report.md").write_text(text, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="EdgeInferenceLab")
    commands = parser.add_subparsers(dest="command", required=True)
    training = commands.add_parser("train")
    training.add_argument("--config", default="configs/paper_aligned.toml")
    training.add_argument("--seed", type=int, default=1)
    training.add_argument("--steps", type=int, default=65536)
    training.add_argument("--out", required=True)
    training.add_argument("--eval-every", type=int, default=4096)
    training.add_argument("--eval-episodes", type=int, default=5)
    training.add_argument("--algorithm", choices=["tp_mppo", "tp_ppo"], default="tp_mppo")
    evaluation = commands.add_parser("evaluate")
    evaluation.add_argument("checkpoint")
    evaluation.add_argument("--seeds", type=int, nargs="+", default=list(range(20000, 20005)))
    evaluation.add_argument("--out", required=True)
    evaluation.add_argument("--no-energy", action="store_true")
    batch = commands.add_parser("suite")
    batch.add_argument("--config", default="configs/paper_aligned.toml")
    batch.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    batch.add_argument("--steps", type=int, default=65536)
    batch.add_argument("--eval-every", type=int, default=4096)
    batch.add_argument("--eval-episodes", type=int, default=5)
    batch.add_argument("--workers", type=int, default=1)
    batch.add_argument("--users", type=int, nargs="+", default=[10, 15, 30])
    batch.add_argument("--nodes", type=int, nargs="+", default=[2, 3, 6])
    batch.add_argument("--out", default="results/g123")
    replay = commands.add_parser("energy-check")
    replay.add_argument("checkpoint")
    replay.add_argument("--seeds", type=int, nargs="+", default=[30000, 30001, 30002])
    replay.add_argument("--out", required=True)
    plotting = commands.add_parser("plot")
    plotting.add_argument("directory")
    integral = commands.add_parser("integrate-power")
    integral.add_argument("csv", help="CSV columns: time_s,power_w")
    ablation = commands.add_parser("benchmark-train", help="Train equal-budget unmasked TP-PPO comparators")
    ablation.add_argument("--reference-suite", default="results/g123_final")
    ablation.add_argument("--out", default="results/g4_training")
    ablation.add_argument("--workers", type=int, default=3)
    frozen = commands.add_parser("benchmark-freeze")
    frozen.add_argument("--reference-suite", default="results/g123_final")
    frozen.add_argument("--seeds", nargs="+", type=int, default=list(range(40000, 40010)))
    frozen.add_argument("--out", default="results/g4/scenarios")
    comparison = commands.add_parser("benchmark")
    comparison.add_argument("--scenarios", default="results/g4/scenarios")
    comparison.add_argument("--reference-suite", default="results/g123_final")
    comparison.add_argument("--ablation-root", default="results/g4_training")
    comparison.add_argument("--methods", nargs="+", default=["tp_mppo", "tp_ppo", "random_masked", "strongest_channel", "balanced", "cap_greedy"])
    comparison.add_argument("--modes", nargs="+", choices=["closed_loop", "fixed"], default=["closed_loop", "fixed"])
    comparison.add_argument("--replicates", nargs="+", type=int, default=[1, 2, 3])
    comparison.add_argument("--workers", type=int, default=3)
    comparison.add_argument("--bootstrap-samples", type=int, default=2000)
    comparison.add_argument("--stats-seed", type=int, default=171)
    comparison.add_argument("--custom", nargs="*", default=[], help="name=module:factory; factory(config,replicate_seed) returns (Policy, training_steps)")
    comparison.add_argument("--allocator", nargs="*", default=[], help="custom_name=module:factory; factory(config) returns a BandwidthAllocator")
    comparison.add_argument("--out", default="results/g4/benchmark")
    summary = commands.add_parser("benchmark-report")
    summary.add_argument("directory")
    replay_run = commands.add_parser("benchmark-replay")
    replay_run.add_argument("directory")
    replay_run.add_argument("--case", required=True)
    replay_run.add_argument("--mode", choices=["closed_loop", "fixed"], required=True)
    replay_run.add_argument("--method", required=True)
    replay_run.add_argument("--replicate", type=int, required=True)
    replay_run.add_argument("--scenario", type=int, required=True)
    replay_run.add_argument("--out", required=True)
    gap = commands.add_parser("g5-run", help="Run the fixed research-gap diagnostic plan without training")
    gap.add_argument("--plan", default="configs/g5.json")
    gap.add_argument("--out", default="results/g5")
    gap.add_argument("--workers", type=int, default=3)
    gap_report = commands.add_parser("g5-report")
    gap_report.add_argument("directory")
    gap_replay = commands.add_parser("g5-replay")
    gap_replay.add_argument("directory")
    gap_replay.add_argument("--case", required=True)
    gap_replay.add_argument("--method", required=True)
    gap_replay.add_argument("--allocator", choices=["tp", "equal_uplink"], required=True)
    gap_replay.add_argument("--replicate", type=int, required=True)
    gap_replay.add_argument("--scenario", type=int, required=True)
    gap_replay.add_argument("--out", required=True)
    args = parser.parse_args()
    if args.command == "train":
        train(load_config(args.config), args.seed, args.steps, args.out,
              eval_every=args.eval_every, eval_episodes=args.eval_episodes, algorithm=args.algorithm)
    elif args.command == "suite":
        suite(args)
    elif args.command == "evaluate":
        agent = MaskedPPO.load(args.checkpoint)
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        results = evaluate(agent.config, agent, args.seeds, energy=not args.no_energy, traces_dir=out / "traces")
        write_json(out / "episodes.json", results)
        write_json(out / "summary.json", mean_metrics(results))
    elif args.command == "energy-check":
        agent = MaskedPPO.load(args.checkpoint)
        write_json(args.out, verify_passive_energy(agent.config, agent, args.seeds))
    elif args.command == "plot":
        from .plotting import plot_suite
        plot_suite(Path(args.directory))
    elif args.command == "integrate-power":
        with Path(args.csv).open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        print(json.dumps({"energy_j": integrate_power_samples([r["time_s"] for r in rows],
                                                              [r["power_w"] for r in rows])}))
    elif args.command == "benchmark-train":
        from .benchmark import train_ablations
        train_ablations(args.reference_suite, args.out, args.workers)
    elif args.command == "benchmark-freeze":
        from .benchmark import freeze_scenarios
        freeze_scenarios(args.reference_suite, args.seeds, args.out)
    elif args.command == "benchmark":
        from .benchmark import run_benchmark
        custom = dict(item.split("=", 1) for item in args.custom)
        assert len(custom) == len(args.custom)
        allocators = dict(item.split("=", 1) for item in args.allocator)
        assert len(allocators) == len(args.allocator)
        run_benchmark(args.scenarios, args.reference_suite, args.ablation_root, args.out,
                      methods=args.methods, modes=args.modes, replicates=args.replicates, workers=args.workers,
                      bootstrap_samples=args.bootstrap_samples, stats_seed=args.stats_seed, custom=custom, allocators=allocators)
    elif args.command == "benchmark-report":
        from .benchmark_stats import summarize_benchmark
        from .benchmark_plotting import plot_benchmark
        summarize_benchmark(args.directory)
        plot_benchmark(args.directory)
    elif args.command == "benchmark-replay":
        from .benchmark import replay_benchmark
        replay_benchmark(args.directory, args.case, args.mode, args.method, args.replicate, args.scenario, args.out)
    elif args.command == "g5-run":
        from .diagnostics import run_g5
        run_g5(args.plan, args.out, args.workers)
    elif args.command == "g5-report":
        from .gap_analysis import analyze_g5
        analyze_g5(Path(args.directory))
    elif args.command == "g5-replay":
        from .diagnostics import replay_g5
        replay_g5(args.directory, args.case, args.method, args.allocator, args.replicate, args.scenario, args.out)


if __name__ == "__main__":
    main()
