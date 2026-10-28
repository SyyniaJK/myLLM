"""Crossed row/column bootstrap over policy replicates and saved scenarios."""

import csv
import json
from pathlib import Path
import numpy as np
from .experiment import write_csv, write_json


METRICS = ("goodput_window", "goodput_ratio_window", "throughput_window", "requests", "energy_j",
           "oom_requests", "rejected", "invalid", "pending", "energy_tail_j", "uplink_j", "downlink_j",
           "gpu_idle_baseline_j", "gpu_prefill_increment_j", "gpu_decode_increment_j")


def score_matrix(rows, metric, replicates, scenarios):
    keys = [(r["replicate"], r["scenario_seed"]) for r in rows]
    assert len(keys) == len(set(keys)), "An evaluation cell occurs exactly once."
    expected = {(r, s) for r in replicates for s in scenarios}
    assert set(keys) == expected, "Every planned policy/scenario cell must be present."
    values = {(r["replicate"], r["scenario_seed"]): r[metric] for r in rows}
    result = np.array([[values[r, s] for s in scenarios] for r in replicates], dtype=float)
    assert np.isfinite(result).all()
    return result


def bootstrap_indices(replicates, scenarios, samples, seed):
    assert replicates > 0 and scenarios > 0 and samples > 0
    rng = np.random.default_rng(seed)
    return (rng.integers(replicates, size=(samples, replicates)),
            rng.integers(scenarios, size=(samples, scenarios)))


def bootstrap_means(matrix, indices):
    rows, columns = indices
    return matrix[rows[:, :, None], columns[:, None, :]].mean(axis=(1, 2))


def summarize_matrix(matrix, indices):
    draws = bootstrap_means(matrix, indices)
    low, high = np.quantile(draws, [0.025, 0.975])
    return {"mean": float(matrix.mean()), "ci_low": float(low), "ci_high": float(high),
            "replicate_sd": float(matrix.mean(axis=1).std(ddof=1)) if len(matrix) > 1 else None}


def energy_per_good(energy, goodput, indices):
    e, g = bootstrap_means(energy, indices), bootstrap_means(goodput, indices)
    undefined = int(np.count_nonzero(g == 0))
    point = float(energy.mean() / goodput.mean()) if goodput.mean() > 0 else None
    # A zero-service sample has undefined efficiency, never an epsilon denominator.
    if undefined:
        low, high = None, None
    else:
        low, high = map(float, np.quantile(e / g, [0.025, 0.975]))
    return {"mean": point, "ci_low": low, "ci_high": high, "undefined_resamples": undefined}


def conditional_latency(rows):
    eligible = [r for r in rows if r["throughput_window"] > 0]
    count = sum(r["throughput_window"] for r in eligible)
    mean = sum(r["e2e_mean_s"] * r["throughput_window"] for r in eligible) / count if count else None
    return {"e2e_mean_s": mean, "latency_completed_requests": count,
            "episodes_without_completed_output": len(rows) - len(eligible)}


def write_pairs(path, rows):
    fields = ["case", "mode", "reference", "method", "metric", "mean", "ci_low", "ci_high", "replicate_sd"]
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def summarize_benchmark(directory):
    directory = Path(directory)
    plan = json.loads((directory / "benchmark.json").read_text(encoding="utf-8"))
    rows = json.loads((directory / "episodes.json").read_text(encoding="utf-8"))
    expected_rows = (len(plan["cases"]) * len(plan["modes"]) * len(plan["methods"])
                     * len(plan["replicates"]) * len(plan["scenario_seeds"]))
    assert len(rows) == expected_rows
    statistics, tables, pairs = [], [], []
    for case, config in plan["cases"].items():
        for mode in plan["modes"]:
            indices = bootstrap_indices(len(plan["replicates"]), len(plan["scenario_seeds"]),
                        plan["bootstrap_samples"], np.random.SeedSequence([
                            plan["stats_seed"], config["users"], config["nodes"], int(mode == "fixed")]))
            all_matrices = {}
            for method in plan["methods"]:
                cells = [r for r in rows if (r["case"], r["mode"], r["method"]) == (case, mode, method)]
                matrices = {metric: score_matrix(cells, metric, plan["replicates"], plan["scenario_seeds"])
                            for metric in METRICS}
                all_matrices[method] = matrices
                table = {"case": case, "mode": mode, "method": method,
                         "replicates": len(plan["replicates"]), "scenarios": len(plan["scenario_seeds"]),
                         "training_steps_per_replicate": cells[0]["training_steps"]}
                for metric, matrix in matrices.items():
                    values = summarize_matrix(matrix, indices)
                    statistics.append({"case": case, "mode": mode, "method": method, "metric": metric, **values})
                    for key, value in values.items():
                        table[f"{metric}_{key}"] = value
                efficiency = energy_per_good(matrices["energy_j"], matrices["goodput_window"], indices)
                table.update({f"joules_per_good_{k}": v for k, v in efficiency.items()})
                table.update(conditional_latency(cells))
                table["policy_decision_ms_mean"] = float(np.mean([r["policy_decision_ms_mean"] for r in cells]))
                table["mean_episode_p95_decision_ms"] = float(np.mean([r["policy_decision_ms_p95"] for r in cells]))
                tables.append(table)
            if "tp_mppo" in all_matrices:
                for method in plan["methods"]:
                    if method != "tp_mppo":
                        for metric in ("goodput_window", "goodput_ratio_window", "energy_j"):
                            difference = all_matrices["tp_mppo"][metric] - all_matrices[method][metric]
                            pairs.append({"case": case, "mode": mode, "reference": "tp_mppo", "method": method,
                                          "metric": metric, **summarize_matrix(difference, indices)})
    write_csv(directory / "statistics.csv", statistics)
    write_csv(directory / "summary.csv", tables)
    write_pairs(directory / "paired_differences.csv", pairs)
    write_json(directory / "summary.json", tables)
    write_json(directory / "paired_differences.json", pairs)
    write_benchmark_report(directory, plan, tables, rows)
    return tables, pairs


def format_interval(row, metric, scale=1):
    mean, low, high = (row[f"{metric}_{k}"] for k in ("mean", "ci_low", "ci_high"))
    if mean is None:
        return "未定义"
    if low is None:
        return f"{mean * scale:.3f}（区间未定义）"
    return f"{mean * scale:.3f} [{low * scale:.3f}, {high * scale:.3f}]"


def write_benchmark_report(directory, plan, tables, rows):
    sections = []
    for case in plan["cases"]:
        for mode in plan["modes"]:
            lines = [f"## {case} / {mode}", "",
                     "| 方法 | 窗口 Goodput [95%区间] | Goodput 比例（%） | 模拟能耗（J） | J/合规请求 [95%区间] | OOM | 非法卸载 |",
                     "|---|---:|---:|---:|---:|---:|---:|"]
            for r in tables:
                if (r["case"], r["mode"]) == (case, mode):
                    lines.append(f"| {r['method']} | {format_interval(r, 'goodput_window')} | "
                        f"{r['goodput_ratio_window_mean'] * 100:.3f} | {r['energy_j_mean']:.3f} | "
                        f"{format_interval(r, 'joules_per_good')} | {r['oom_requests_mean']:.3f} | {r['invalid_mean']:.3f} |")
            sections.append("\n".join(lines))
    text = f"""# G4：统一算法对比与可复现评估

本次实际完成 {len(rows)} 次测试评估：{len(plan['cases'])} 个规模配置 × {len(plan['modes'])} 种负载模式 × {len(plan['methods'])} 种方法 × {len(plan['replicates'])} 个策略重复 × {len(plan['scenario_seeds'])} 个保存场景。

## 算法与训练预算

- `tp_mppo`：复用 G1–G3 的带掩码 PPO 卸载策略。
- `tp_ppo`：同一网络、超参数、奖励、资源分配和交互预算，训练和评估均取消动作掩码。
- `random_masked`：在合法动作（含拒绝）中均匀随机。
- `strongest_channel`：为每个请求选择最强上行信道的空闲节点。
- `balanced`：依次选择本批已分配请求最少的空闲节点，信道用于打破平局。
- `cap_greedy`：按声明输出 cap 排序，选择加入请求后峰值显存最小且不超容量的节点。

后四者是本平台的基础对照，未冒用原论文 Rewardless 或引用文献中的 Heuristic 名称。自定义方法的 factory 记录在 benchmark.json。
学习方法的每个 case/seed 预算在运行前按训练元数据核对相等；启发式没有训练步骤。使用最终 checkpoint，不从测试结果中挑选最好 checkpoint。

## 两种负载模式

固定场景已保存上下行信道、部署随机种子和完整外生请求字段（包含环境内部使用的实际输出）。这些未来字段不进入策略观测。

- **fixed**：每用户每时隙最多一个外生请求，提供概率为 1−exp(−τ/mean_think_s)。请求可以与该用户较早的请求重叠；不设跨时隙 AP 队列。节点忙碌时由动作和共同环境规则决定拒绝。每个方法逐字段核对完全相同的外部请求，不因完成时间改变输入。
- **closed_loop**：原论文的单用户单在途请求和完成后 thinking time。各方法共享信道与每用户随机规则，实际后续到达会受各自服务速度影响，不声称请求轨迹相同。

学习策略均在闭环模式训练，fixed 是冻结策略的外生负载迁移评估；两类结果分别报告，不混为同分布训练测试。

## 指标与统计契约

主要 Goodput、Throughput 和能耗均按同一 [0,H] 窗口结算，窗口末尾未完成请求单列 pending。`goodput_cohort` 和尾部能耗另存，不混用于能效比。
能效为平均窗口能耗/平均窗口合规请求数；零服务时未定义，不用小常数替代分母。平均 E2E 只针对窗口内已返回输出，summary.csv 同时提供对应请求数。

策略重复数为 {len(plan['replicates'])}，场景数为 {len(plan['scenario_seeds'])}。对学习方法前者是训练 seed；确定性启发式的重复行相同，不把它们当作独立训练样本。
报告均值、策略重复间 SD 和名义 95% crossed-bootstrap 百分位区间：分别重采样策略行与场景列，{plan['bootstrap_samples']} 次，统计随机种子 {plan['stats_seed']}。
方法差使用同一批配对行列重采样，见 paired_differences.csv（TP-MPPO 减其他方法）。区间是有限样本的探索性不确定性描述，不是同时覆盖所有比较的保证，不据此标显著性星号。
物理指标和动作可精确重放；策略决策耗时是 CPU 墙钟诊断量，只包括 act()，不包括环境推进或带宽求解，不作为精确重放字段。

统计方法参考 [Owen: The pigeonhole bootstrap](https://arxiv.org/abs/0712.1111) 的行列重采样思想；强化学习评估原则参考 [Agarwal et al., NeurIPS 2021](https://arxiv.org/abs/2108.13264)。本实现没有把不同请求当成独立训练重复，也没有把所有 case 混成单一排名。

{chr(10).join(sections)}

## 可复现交付

- benchmark.json：算法、输入位置、场景/策略 seed、运行库版本、统计参数和边界定义。
- episodes.csv / episodes.json：所有计划单元的原始逐场景指标。
- summary.csv / statistics.csv：方法汇总和逐指标区间。
- paired_differences.csv：配对方法差及区间。
- episodes/：每次运行的动作/reward，首场景另保存逐请求、逐批次与功率区间，以及重放检查。
- `benchmark-replay`：按 case/mode/method/replicate/scenario 重建任意保存运行，比较动作、reward 和物理指标。

当前功率仍是 G3 的相位估算参数，能耗列不代表真实 GPU 的测量结果。资源和功率模型对所有方法一致。
"""
    (Path(directory) / "report.md").write_text(text, encoding="utf-8")
