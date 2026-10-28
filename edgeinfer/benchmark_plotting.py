"""G4 figures from saved complete evaluations; see docs/g4_contract.md."""

import csv
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


LABELS = {"tp_mppo": "TP-MPPO", "tp_ppo": "TP-PPO", "random_masked": "Random",
          "strongest_channel": "Channel", "balanced": "Balanced", "cap_greedy": "Cap-greedy"}
COLORS = ("#0072B2", "#56B4E9", "#999999", "#E69F00", "#009E73", "#CC79A7")
MARKERS = ("o", "s", "D", "^", "v", "P")


def save_figure(fig, filename):
    fig.savefig(filename.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(filename.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(filename.with_suffix(".png"), dpi=600, bbox_inches="tight")
    plt.close(fig)


def error(row, metric, scale=1):
    mean = row[f"{metric}_mean"]
    return np.array([[mean - row[f"{metric}_ci_low"]], [row[f"{metric}_ci_high"] - mean]]) * scale


def plot_benchmark(directory):
    directory = Path(directory)
    plan = json.loads((directory / "benchmark.json").read_text(encoding="utf-8"))
    rows = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    pairs = json.loads((directory / "paired_differences.json").read_text(encoding="utf-8"))
    out = directory / "figures"
    out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
                         "font.size": 8, "svg.fonttype": "none", "pdf.fonttype": 42,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "legend.frameon": False, "axes.linewidth": 0.8})
    methods, modes = plan["methods"], plan["modes"]
    labels = {m: LABELS.get(m, m) for m in methods}
    colors = {m: COLORS[i % len(COLORS)] for i, m in enumerate(methods)}
    markers = {m: MARKERS[i % len(MARKERS)] for i, m in enumerate(methods)}
    case = "default" if "default" in plan["cases"] else next(iter(plan["cases"]))
    lookup = {(r["case"], r["mode"], r["method"]): r for r in rows}
    handles = [Line2D([], [], color=colors[m], marker=markers[m], ls="", label=labels[m]) for m in methods]
    fig, axes = plt.subplots(len(modes), 2, squeeze=False, figsize=(7.15, 2.7 * len(modes)), layout="constrained")
    for i, mode in enumerate(modes):
        values = [lookup[case, mode, m] for m in methods]
        ax = axes[i, 0]
        means = [r["goodput_window_mean"] for r in values]
        errors = np.concatenate([error(r, "goodput_window") for r in values], axis=1)
        ax.bar(np.arange(len(methods)), means, color=[colors[m] for m in methods], yerr=errors,
               capsize=2, error_kw={"elinewidth": 0.7})
        ax.set(xticks=np.arange(len(methods)), xticklabels=[labels[m] for m in methods], ylabel="Good requests in [0,H]")
        ax.tick_params(axis="x", labelrotation=35, labelsize=7)
        ax.set_title(f"{mode} · service", fontsize=9)
        ax = axes[i, 1]
        for method, row in zip(methods, values):
            ax.errorbar(row["goodput_window_mean"], row["energy_j_mean"] / 1000,
                        xerr=error(row, "goodput_window"), yerr=error(row, "energy_j", 1 / 1000),
                        marker=markers[method], color=colors[method], capsize=2, ms=4, elinewidth=0.7, ls="")
        ax.set(xlabel="Good requests in [0,H]", ylabel="Modelled energy (kJ)")
        ax.set_title(f"{mode} · service–energy", fontsize=9)
    fig.suptitle(f"{case} · mean and nominal 95% crossed-bootstrap intervals", fontsize=9)
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.01), ncol=3, fontsize=8)
    save_figure(fig, out / "performance")

    if "default" in plan["cases"]:
        fig, axes = plt.subplots(len(modes), 2, squeeze=False, figsize=(7.15, 2.5 * len(modes)), layout="constrained")
        for i, mode in enumerate(modes):
            for ax, prefix, field, xlabel in zip(axes[i], ("users_", "nodes_"), ("users", "nodes"), ("Users", "Edge nodes")):
                cases = [c for c in plan["cases"] if c == "default" or c.startswith(prefix)]
                cases.sort(key=lambda name: plan["cases"][name][field])
                for method in methods:
                    x = [plan["cases"][c][field] for c in cases]
                    data = [lookup[c, mode, method] for c in cases]
                    y = [r["goodput_ratio_window_mean"] * 100 for r in data]
                    bars = np.concatenate([error(r, "goodput_ratio_window", 100) for r in data], axis=1)
                    ax.errorbar(x, y, yerr=bars, color=colors[method], marker=markers[method],
                                capsize=2, ms=3, lw=1, elinewidth=0.6)
                ax.set(xlabel=xlabel, ylabel="Window Goodput ratio (%)", xticks=x)
                ax.set_title(mode, fontsize=9)
        fig.suptitle("Scale checks · mean and nominal 95% crossed-bootstrap intervals", fontsize=9)
        fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.01), ncol=3, fontsize=8)
        save_figure(fig, out / "scaling")

    comparisons = [m for m in methods if m != "tp_mppo"]
    if "tp_mppo" in methods and comparisons:
        fig, axes = plt.subplots(1, len(modes), squeeze=False, figsize=(7.15, 2.8), layout="constrained")
        paired = {(r["case"], r["mode"], r["method"], r["metric"]): r for r in pairs}
        for ax, mode in zip(axes[0], modes):
            ax.axvline(0, color="#8B9299", ls="--", lw=0.8)
            for y, method in enumerate(comparisons):
                r = paired[case, mode, method, "goodput_window"]
                ax.errorbar(r["mean"], y, xerr=[[r["mean"] - r["ci_low"]], [r["ci_high"] - r["mean"]]],
                            marker=markers[method], color=colors[method], capsize=3, ms=4, ls="")
            ax.set(yticks=np.arange(len(comparisons)), yticklabels=[labels[m] for m in comparisons],
                   xlabel="TP-MPPO minus comparator: good requests")
            ax.invert_yaxis()
            ax.set_title(mode, fontsize=9)
        fig.suptitle(f"{case} · paired differences and nominal 95% intervals", fontsize=9)
        save_figure(fig, out / "paired")

    if {"tp_mppo", "tp_ppo"} <= set(methods):
        fig, axes = plt.subplots(1, 2, figsize=(7.15, 2.8), layout="constrained")
        for method, root in (("tp_mppo", directory / plan["reference_root"]),
                             ("tp_ppo", directory / plan["ablation_root"])):
            curves = []
            for replicate in plan["replicates"]:
                with (root / case / f"seed_{replicate}" / "learning_curve.csv").open(encoding="utf-8", newline="") as handle:
                    curves.append(list(csv.DictReader(handle)))
            steps = np.array([float(r["step"]) for r in curves[0]])
            assert all([r["step"] for r in c] == [r["step"] for r in curves[0]] for c in curves)
            for ax, metric, scale, ylabel in zip(axes, ("reward", "goodput_ratio"), (1, 100),
                                                ("Validation episode reward", "Validation cohort Goodput (%)")):
                values = np.array([[float(r[metric]) for r in c] for c in curves]) * scale
                mean = values.mean(axis=0)
                sd = values.std(axis=0, ddof=1) if len(values) > 1 else np.zeros_like(mean)
                ax.plot(steps / 1000, mean, color=colors[method], label=labels[method], lw=1.4)
                ax.fill_between(steps / 1000, mean - sd, mean + sd, color=colors[method], alpha=0.15, linewidth=0)
                ax.set(xlabel="Training steps (×1000)", ylabel=ylabel)
        axes[0].legend(fontsize=8)
        fig.suptitle(f"{case} · equal-budget training · shading: replicate SD", fontsize=9)
        save_figure(fig, out / "learning_comparison")
