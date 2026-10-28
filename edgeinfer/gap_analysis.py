"""Source-backed G5 summaries and diagnostic figures; no fitted optimization model."""

import csv
from collections import defaultdict
import json
from pathlib import Path
import numpy as np
from .benchmark_stats import bootstrap_indices, score_matrix, summarize_matrix
from .experiment import write_csv, write_json


METRICS = ("requests", "goodput_window", "goodput_ratio_window", "energy_j", "fixed_j", "variable_window_j",
           "radio_window_j", "without_rf_window_j", "gpu_increment_window_j", "gpu_increment_full_j",
           "gpu_increment_tail_j", "variable_full_j", "gpu_solo_full_j", "gpu_utilization",
           "batch_upload_sum_s", "batch_wait_sum_s", "oom_requests", "rejected", "pending")
TEXT = {"experiment", "case", "method", "allocator", "trace_key", "choices", "mode"}


def read_table(path):
    with Path(path).open(encoding="utf-8", newline="") as handle:
        return [{k: (v if k in TEXT else v == "True" if k == "future_information"
                     else float(v) if v else None) for k, v in row.items()} for row in csv.DictReader(handle)]


def summarize_rows(rows, plan, metrics):
    grouped = defaultdict(list)
    for r in rows:
        grouped[r["case"], r["method"], r["allocator"]].append(r)
    cases = sorted({r["case"] for r in rows})
    results = []
    for (case, method, allocator), cells in sorted(grouped.items()):
        reps = sorted({r["replicate"] for r in cells})
        assert {r["scenario_seed"] for r in cells} == set(plan["seeds"])
        indices = bootstrap_indices(len(reps), len(plan["seeds"]), plan["bootstrap_samples"],
                    np.random.SeedSequence([plan["stats_seed"], cases.index(case)]))
        row = {"case": case, "method": method, "allocator": allocator,
               "replicates": len(reps), "scenarios": len(plan["seeds"])}
        row.update({k: cells[0][k] for k in ("mean_think_s", "attenuation_db", "slo_s", "memory_gb") if k in cells[0]})
        for metric in metrics:
            matrix = score_matrix(cells, metric, reps, plan["seeds"])
            row.update({f"{metric}_{k}": v for k, v in summarize_matrix(matrix, indices).items()})
        results.append(row)
    return results


def paired_wireless(rows, plan):
    results = []
    for think in plan["mean_think_s"]:
        for method in plan["methods"]:
            for allocator in plan["allocators"]:
                base = [r for r in rows if (r["mean_think_s"], r["method"], r["allocator"], r["attenuation_db"])
                        == (think, method, allocator, 0)]
                for db in plan["attenuation_db"][1:]:
                    changed = [r for r in rows if (r["mean_think_s"], r["method"], r["allocator"], r["attenuation_db"])
                               == (think, method, allocator, db)]
                    indices = bootstrap_indices(len(plan["replicates"]), len(plan["seeds"]), plan["bootstrap_samples"],
                            np.random.SeedSequence([plan["stats_seed"], int(think*10), db]))
                    for metric in ("goodput_window", "gpu_increment_full_j", "gpu_increment_window_j",
                                   "gpu_increment_tail_j", "without_rf_window_j"):
                        difference = (score_matrix(changed, metric, plan["replicates"], plan["seeds"])
                                      - score_matrix(base, metric, plan["replicates"], plan["seeds"]))
                        results.append({"mean_think_s": think, "method": method, "allocator": allocator,
                            "attenuation_db": db, "metric": metric, **summarize_matrix(difference, indices)})
    return results


def analyze_g5(directory):
    directory = Path(directory)
    plan = json.loads((directory / "plan.json").read_text(encoding="utf-8"))
    scan = json.loads((directory / "scan.json").read_text(encoding="utf-8"))
    composition, length, exact = [read_table(directory / f"{name}.csv") for name in ("composition", "length", "exact")]
    expected = {(think, db, allocator, method, rep, seed) for think in plan["mean_think_s"]
        for db in plan["attenuation_db"] for allocator in plan["allocators"] for method in plan["methods"]
        for rep in plan["replicates"] for seed in plan["seeds"]}
    actual = {(r["mean_think_s"], r["attenuation_db"], r["allocator"], r["method"], r["replicate"], r["scenario_seed"]) for r in scan}
    assert len(scan) == len(actual) and actual == expected
    assert len(composition) == len(plan["seeds"]) * len(plan["snapshot_slo_s"]) * len(plan["snapshot_attenuation_db"]) * 2 * len(plan["allocators"])
    assert len(length) == len(plan["seeds"]) * len(plan["memory_gb"]) * len(plan["estimators"])
    assert len(exact) == len(plan["seeds"]) * len(plan["snapshot_slo_s"]) * len(plan["snapshot_attenuation_db"]) * (len(plan["methods"])*len(plan["replicates"])+2)
    for r in exact:
        r["allocator"] = "tp"
    tables = {
        "scan_summary": summarize_rows(scan, plan, METRICS),
        "composition_summary": summarize_rows(composition, plan, METRICS),
        "length_summary": summarize_rows(length, plan, (*METRICS, "length_mae_tokens", "underestimated_requests")),
        "exact_summary": summarize_rows(exact, plan, (*METRICS, "service_gap", "same_service_energy_gap_j", "charged_variable_full_j")),
        "wireless_differences": paired_wireless(scan, plan),
    }
    for name, rows in tables.items():
        write_csv(directory / f"{name}.csv", rows)
        write_json(directory / f"{name}.json", rows)
    write_report(directory, plan, scan, composition, length, exact, tables)
    plot_g5(directory, plan, tables)
    return tables


def write_report(directory, plan, scan, composition, length, exact, tables):
    decomposition = read_table(directory / "energy_decomposition.csv")
    nominal = [r for r in decomposition if (r["case"], r["mode"], r["method"]) == ("default", "closed_loop", "tp_mppo")]
    mean = lambda rows, key: float(np.mean([r[key] for r in rows]))
    fixed_fraction = mean(nominal, "fixed_j") / mean(nominal, "energy_j")
    radio_fraction = mean(nominal, "radio_j") / mean(nominal, "energy_j")
    differences = tables["wireless_differences"]
    max_gp = max(differences, key=lambda r: abs(r["mean"]) if r["metric"] == "goodput_window" else -1)
    max_gpu = max(differences, key=lambda r: abs(r["mean"]) if r["metric"] == "gpu_increment_full_j" else -1)
    controlled = defaultdict(dict)
    for r in composition:
        controlled[r["case"], r["method"], r["scenario_seed"]][r["allocator"]] = r
    same_batch_diffs = [a["equal_uplink"]["gpu_increment_full_j"] - a["tp"]["gpu_increment_full_j"] for a in controlled.values()]
    assert all(x == 0 for x in same_batch_diffs)
    sections = []
    for memory in plan["memory_gb"]:
        lines = [f"### {memory:g} GB", "", "| 长度输入 | 未来信息 | Goodput | OOM 请求 | 拒绝 | 平均绝对误差/token |", "|---|---|---:|---:|---:|---:|"]
        for mode in plan["estimators"]:
            cells = [r for r in length if (r["memory_gb"], r["method"]) == (memory, mode)]
            lines.append(f"| {mode} | {'是' if cells[0]['future_information'] else '否'} | {mean(cells,'goodput_window'):.2f} | {mean(cells,'oom_requests'):.2f} | {mean(cells,'rejected'):.2f} | {mean(cells,'length_mae_tokens'):.2f} |")
        sections.append("\n".join(lines))
    exact_sections = []
    for case in sorted({r["case"] for r in exact}):
        lines = [f"### {case}", "", "| 方法 | Goodput | 距条件 oracle 的服务差距 | 同服务数完整变量能耗差/J |", "|---|---:|---:|---:|"]
        for method in [*plan["methods"], "cap_exact", "oracle_exact"]:
            cells = [r for r in exact if (r["case"], r["method"]) == (case, method)]
            lines.append(f"| {method} | {mean(cells,'goodput_window'):.3f} | {mean(cells,'service_gap'):.3f} | {mean(cells,'same_service_energy_gap_j'):.3f} |")
        exact_sections.append("\n".join(lines))
    text = f"""# G5：研究问题诊断结果

## Material Passport

- 类型：模型内数值诊断；新神经网络训练次数为 0。
- 执行代码版本：`{plan['code_commit']}`，运行配置 `plan.json`，基础契约 `docs/g5_contract.md`。
- 数据：{len(scan)} 次完整二维扫描、{len(length)} 次长度误差评估、{len(composition)} 个批次组成控制快照、{len(exact)} 个方法/精确计划快照；另分解 1800 条既有 G4 记录。
- 证据层级：模型恒等式与合成干预；未完成目标 GPU/无线网络的物理校准。重放状态另见 verification.json。

## G5-1：能耗模型的可靠性边界

记账单位和固定基线契约已核对。默认相位功率、理想有效算力、8 bits/token 和信道参数没有真实目标设备的标定依据。相位常功率积分是当前模型内的估算，不是预测真实 GPU 功率曲线的已验证模型。

[Zeus](https://ml.energy/zeus/measure/) 提供硬件测量窗口；[TokenPowerBench](https://github.com/chenxuniu/TokenPowerBench) 的测量流程用于对照定义。这里没有运行它们采集硬件数据。详细缺口见 model_audit.json；这部分完成了审计，尚未完成硬件校准。

## G5-2：固定项与变量项

G4 默认闭环 TP-MPPO 的固定基线占平均总能耗 **{100*fixed_fraction:.3f}%**，射频占 **{100*radio_fraction:.6f}%**；固定基线为 {mean(nominal,'fixed_j'):.0f} J。将全部变量项消除的代数上限是 {100*(1-fixed_fraction):.3f}%，不代表在相同服务量下能实现该比例。

未引入节点休眠、DVFS 或模型切换。当前调度对固定项的导数为零；这不能通过给 PPO 奖励加权改变。不同方法降低总能耗时可能只是少服务，必须结合 Goodput 和同服务数前沿判断。

## G5-3：无线与提供负载

完整扫描覆盖三个衰减、三个提供负载、六种方法、两个上行分配器、三个策略重复、五个场景。提供负载轴不是 GPU 时钟或 GPU 实测利用率。PPO 是 nominal 策略的迁移诊断，不能把新条件下的差距直接归因于算法本身不可优化。

相对同负载 0 dB，最大绝对配对 Goodput 均值变化为 **{max_gp['mean']:.3f}** 个请求（think={max_gp['mean_think_s']:g}, {max_gp['attenuation_db']} dB, {max_gp['method']}, {max_gp['allocator']}）；其名义区间为 [{max_gp['ci_low']:.3f}, {max_gp['ci_high']:.3f}]。
最大绝对完整 GPU 增量能耗配对均值变化为 **{max_gpu['mean']:.3f} J**（think={max_gpu['mean_think_s']:g}, {max_gpu['attenuation_db']} dB, {max_gpu['method']}, {max_gpu['allocator']}）。全部配对区间、窗口与尾部的区别保存在 wireless_differences.csv，没有只保留表现有利的格点。

射频记账置零后，`without_rf_window_j` 仍可能随无线条件变化；这属于时序、接纳和批次组成的间接作用。对于相同请求及放置的 {len(same_batch_diffs)} 对控制，上行分配改变后的**完整 GPU 增量能耗差全部精确为零**。因此不能声称“GPU 多等上传就会多消耗动态计算能量”；当前模型只能通过改变后续接纳/组成，或窗口截断，产生这类 GPU 能耗差。

## G5-4：SLO 与批次组成

grouped/mixed 控制保持相同六请求、总输入输出量、节点数量和每批三请求。组成可通过 padding 和最长 decode 改变完整 GPU 工作，而非仅改变开始时刻。逐格服务数、完整增量和窗口/尾部见 composition.csv；图中 SLO 曲线展示 0 dB，40 dB 对照完整保留在数据与汇总。

## G5-5：长度估计的可行性风险

下面各值为五个新场景均值。前三种为 cap 代理输入；后三种包含未来信息。这里没有训练预测器，误差代理不能解释为真实预测模型的准确率。OOM 沿用上传后解析失败，不包含真实 GPU 部分执行与恢复开销。

{chr(10).join(sections)}

## G5-6：小规模条件优化空间

每个快照完整枚举 4096 个离散放置，30 个快照共 122880 个不同放置（真实成本与 cap 成本分别评价）。服务差距非负；同服务数能耗差参照真实成本前沿。固定基线相同；表中能耗差对完整变量成本计费，避免用拖到窗口外伪造节能。

{chr(10).join(exact_sections)}

精确性仅限单次放置及固定 TP P2/P3；未优化跨时隙排队、动态批处理或其他连续资源规则。快照真实输出等于 cap，因而此处 cap_exact 与 oracle_exact 的信息优势相同；长度未知的风险由 G5-5 单独检验。不能把该前沿称为完整系统全局最优。

## 问题验证与下一步

| 候选问题 | 本轮可证实的范围 | 研究判断 |
|---|---|---|
| 无线等待影响 GPU 成本 | 同批次完整成本不变；系统级接纳/组成和窗口边界可变化 | 保留跨时隙接纳/组成诊断；不以“等待本身节省动态能耗”为论点 |
| 输出长度误差 | cap 代理与未来信息控制可比较可行性和 OOM 风险 | 属已有研究问题；须有机制区别与校准证据才可形成主创新 |
| 固定基线不可控 | 模型内严格成立，且占主要能耗 | 当前模型的重要边界；休眠/激活已有先例，不直接视为创新 |
| 贪心的条件差距 | 以每个 SLO/衰减的单次精确枚举衡量 | 服务差与同服务能耗差分开判断；不外推成全时域最优差距 |

本轮没有证明学术新颖性。七篇论文与新增摘要中的重合风险沿用用户讨论，未把“这些材料没有处理”改写为“领域无人研究”。若继续推进，先对有差距的具体条件校准有效推理性能/功率，并核查相同机制的最近工作；不能因为极端链路压力下出现差距就宣布现实场景价值。

## 统计与解释检查

全部格点保留，按负载/SLO/衰减分层。名义 95% 交叉 bootstrap（2000 次）为探索性描述，不做多重比较显著性宣称；确定性重复不扩大有效训练样本数。原始成功、失败、拒绝和 pending 均保留，没有只统计成功请求来推断总体服务。

11 项方法学检查：Simpson（不混条件排名）；生态谬误（不从聚合推个体）；Berkson（不筛成功样本）；collider（不按服务成功过滤来估计干预效果）；基率忽略（保留 offered 分母）；均值回归（无极端 seed 挑选）；幸存偏差（保留全部失败）；多重寻找（完整网格且不标显著）；分析分叉（运行前保存配置/契约）；因果误读（仅解释模型内受控干预）；反向因果（参数干预先于执行）。没有真实系统因果/普适性声明。

## 文件与复现

`plan.json`、`inputs/`、`trajectories/` 为本轮输入/动作记录；scan/length/composition/exact CSV 为原始结果，*_summary 与 wireless_differences 为统计；exact_frontier.csv 为所有可达成功数的条件前沿。所有图来自这些原始表。用 `g5-replay` 重放主扫描，用 `g5-report` 重建统计及图。
"""
    (directory / "report.md").write_text(text, encoding="utf-8")


def plot_g5(directory, plan, tables):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from .benchmark_plotting import LABELS, COLORS
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": 8, "svg.fonttype": "none", "pdf.fonttype": 42, "legend.frameon": False,
        "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.8})
    out = directory / "figures"
    out.mkdir(exist_ok=True)
    def save(fig, name):
        fig.savefig(out / f"{name}.svg", bbox_inches="tight")
        fig.savefig(out / f"{name}.pdf", bbox_inches="tight")
        fig.savefig(out / f"{name}.png", dpi=600, bbox_inches="tight")
        plt.close(fig)
    def interval(ax, x, rows, metric, **kwargs):
        y = np.array([r[f"{metric}_mean"] for r in rows])
        low = np.array([r[f"{metric}_ci_low"] for r in rows])
        high = np.array([r[f"{metric}_ci_high"] for r in rows])
        ax.plot(x, y, **kwargs)
        ax.vlines(x, low, high, color=kwargs["color"], lw=0.8)
    raw = read_table(directory / "energy_decomposition.csv")
    nominal = [r for r in raw if r["case"] == "default" and r["mode"] == "closed_loop"]
    fig, ax = plt.subplots(figsize=(7.15, 2.8), layout="constrained")
    bottom = np.zeros(len(plan["methods"]))
    for key, color, label in (("fixed_j", "#999999", "Fixed idle"), ("gpu_increment_j", "#009E73", "GPU increment"),
                               ("radio_j", "#E69F00", "Radio")):
        values = np.array([np.mean([r[key] for r in nominal if r["method"] == m]) for m in plan["methods"]]) / 1000
        ax.bar(np.arange(len(bottom)), values, bottom=bottom, color=color, label=label)
        bottom += values
    ax.set(xticks=range(len(bottom)), xticklabels=[LABELS[m] for m in plan["methods"]], ylabel="Modelled energy (kJ)",
           title="G4 default closed loop · mean window energy decomposition")
    ax.tick_params(axis="x", labelrotation=20)
    fig.legend(*ax.get_legend_handles_labels(), loc="outside lower center", ncol=3)
    save(fig, "energy_decomposition")

    scan = [r for r in tables["scan_summary"] if r["allocator"] == "tp"]
    fig, axes = plt.subplots(2, 6, figsize=(11.5, 4.6), layout="constrained")
    for i, (metric, scale, title, cmap) in enumerate((("goodput_ratio_window",100,"Window Goodput (%)","viridis"),
                     ("gpu_increment_full_j",1,"Fully charged GPU increment (J)","cividis"))):
        values = np.array([r[f"{metric}_mean"]*scale for r in scan])
        for j, method in enumerate(plan["methods"]):
            image = np.array([[next(r for r in scan if (r["method"],r["mean_think_s"],r["attenuation_db"])
                           == (method,think,db))[f"{metric}_mean"]*scale for db in plan["attenuation_db"]]
                           for think in plan["mean_think_s"]])
            ax = axes[i,j]
            im = ax.imshow(image, vmin=values.min(), vmax=values.max(), cmap=cmap, aspect="auto")
            ax.set(xticks=range(3), xticklabels=plan["attenuation_db"], yticks=range(3), yticklabels=["Low","Nominal","High"])
            if i == 0:
                ax.set_title(LABELS[method], fontsize=9)
            if i == 1:
                ax.set_xlabel("Extra attenuation (dB)")
            if j == 0:
                ax.set_ylabel("Offered load")
            for a in range(3):
                for b in range(3):
                    ax.text(b,a,f"{image[a,b]:.1f}" if i==0 else f"{image[a,b]:.0f}",ha="center",va="center",fontsize=6,
                            color="white" if image[a,b] < values.min()+0.55*(values.max()-values.min()) else "black")
        fig.colorbar(im, ax=axes[i,:], shrink=0.8, label=title)
    fig.suptitle("Fixed-input transfer diagnostics · TP uplink · means over saved scenarios / policy seeds")
    save(fig, "wireless_load")

    fig, axes = plt.subplots(2,2,figsize=(8,6.2), layout="constrained")
    for method,color in (("grouped",COLORS[0]),("mixed",COLORS[3])):
        for allocator,style in (("tp","-"),("equal_uplink","--")):
            cells = sorted([r for r in tables["composition_summary"] if (r["method"],r["allocator"],r["attenuation_db"])
                            == (method,allocator,0)], key=lambda r:r["slo_s"])
            for ax,metric in zip(axes[0],("goodput_window","gpu_increment_full_j")):
                interval(ax,[r["slo_s"] for r in cells],cells,metric,color=color,marker="o",ls=style,label=f"{method} / {allocator}")
    axes[0,0].set(xlabel="SLO (s)",ylabel="Good requests / six offered",title="Same requests, different grouping · 0 dB")
    axes[0,1].set(xlabel="SLO (s)",ylabel="Full GPU increment (J)",title="Whole-batch cost vs upload timing")
    axes[0,1].legend(fontsize=6)
    for memory,color in zip(plan["memory_gb"],(COLORS[0],COLORS[3])):
        cells = [next(r for r in tables["length_summary"] if (r["memory_gb"],r["method"])==(memory,m)) for m in plan["estimators"]]
        for ax,metric in zip(axes[1],("goodput_window","oom_requests")):
            interval(ax,np.arange(len(cells)),cells,metric,color=color,marker="s",label=f"{memory:g} GB")
    for ax in axes[1]:
        ax.set(xticks=range(6),xticklabels=["Cap","Mean","Half-cap","Oracle","Half-true","1.25-true"])
        ax.tick_params(axis="x",labelrotation=35,labelsize=7)
        ax.axvline(2.5,color="#999999",ls=":",lw=0.8)
        ax.legend(fontsize=7)
    axes[1,0].set(ylabel="Good requests / 200 s",title="Length proxies · right three use future truth")
    axes[1,1].set(ylabel="OOM requests / 200 s",title="Nominal 0 dB · high offered load")
    fig.suptitle("Controlled diagnostics · means and nominal 95% bootstrap intervals")
    save(fig,"composition_length")

    fig,axes=plt.subplots(2,3,figsize=(10,6),layout="constrained")
    methods=[*plan["methods"],"cap_exact","oracle_exact"]
    for i,db in enumerate(plan["snapshot_attenuation_db"]):
        for j,slo in enumerate(plan["snapshot_slo_s"]):
            cells=[next(r for r in tables["exact_summary"] if (r["method"],r["slo_s"],r["attenuation_db"])==(m,slo,db)) for m in methods]
            ax=axes[i,j]
            y=[r["service_gap_mean"] for r in cells]
            ax.bar(range(len(cells)),y,color=[*COLORS,"#666666","#222222"])
            ax.vlines(range(len(cells)),[r["service_gap_ci_low"] for r in cells],[r["service_gap_ci_high"] for r in cells],color="black",lw=0.8)
            ax.set(xticks=range(len(cells)),xticklabels=[LABELS.get(m,m.replace("_","-")) for m in methods],
                   ylabel="Good-request gap",title=f"SLO {slo:g} s / extra {db:g} dB")
            ax.tick_params(axis="x",labelrotation=45,labelsize=6)
    fig.suptitle("Conditional single-decision gap · all 4096 placements · means and nominal 95% intervals")
    save(fig,"exact_gap")
