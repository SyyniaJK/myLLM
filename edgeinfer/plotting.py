"""Validation figures from actual suite CSV/JSON only; see docs/figure_contract.md."""

import csv
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def read_csv(path):
    with Path(path).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def save(fig, path):
    fig.savefig(path.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), dpi=600, bbox_inches="tight")
    plt.close(fig)


def plot_suite(directory):
    directory = Path(directory)
    output = directory / "figures"
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
                         "font.size": 8, "svg.fonttype": "none", "pdf.fonttype": 42,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "legend.frameon": False, "axes.linewidth": 0.8})
    paths = sorted((directory / "default").glob("seed_*/learning_curve.csv"))
    curves = [read_csv(path) for path in paths]
    steps = np.array([float(r["step"]) for r in curves[0]])
    assert all([r["step"] for r in c] == [r["step"] for r in curves[0]] for c in curves)
    fig, axes = plt.subplots(1, 3, figsize=(7.05, 2.5), layout="constrained")
    for ax, key, ylabel, multiplier, letter in zip(axes,
            ("reward", "goodput_ratio", "oom_requests"),
            ("Episode reward", "Goodput ratio (%)", "OOM requests / episode"),
            (1, 100, 1), ("a", "b", "c")):
        values = np.array([[float(r[key]) for r in c] for c in curves]) * multiplier
        mean = values.mean(axis=0)
        sd = values.std(axis=0, ddof=1) if len(curves) > 1 else np.zeros_like(mean)
        ax.fill_between(steps / 1000, mean - sd, mean + sd, color="#3778A4", alpha=0.16, linewidth=0)
        ax.plot(steps / 1000, mean, color="#27658C", lw=1.5)
        ax.set(xlabel="Training steps (×1000)", ylabel=ylabel)
        ax.set_title(letter, loc="left", fontweight="bold")
        if key == "oom_requests" and not values.any():
            ax.set(ylim=(0, 1), yticks=[0])
            ax.text(0.5, 0.7, "No OOM observed", transform=ax.transAxes,
                    ha="center", color="#58636B", fontsize=8)
    fig.suptitle(f"TP-MPPO · {len(curves)} training seeds · shading: sample SD", fontsize=9)
    save(fig, output / "learning")

    rows = read_csv(directory / "summary.csv")
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.6), layout="constrained")
    for ax, prefix, field, xlabel in zip(axes, ("users_", "nodes_"), ("users", "nodes"), ("Users", "Edge nodes")):
        selected = [r for r in rows if r["case"] == "default" or r["case"].startswith(prefix)]
        x = sorted(set(int(r[field]) for r in selected))
        groups = [np.array([float(r["goodput_ratio"]) * 100 for r in selected if int(r[field]) == v]) for v in x]
        means = [g.mean() for g in groups]
        sd = [g.std(ddof=1) if len(g) > 1 else 0 for g in groups]
        ax.errorbar(x, means, yerr=sd, marker="o", capsize=3, color="#27658C", lw=1.3)
        ax.set(xlabel=xlabel, ylabel="Goodput ratio (%)", xticks=x)
    fig.suptitle("Scale checks · mean ± sample SD across training seeds", fontsize=9)
    save(fig, output / "scaling")

    episodes = []
    for path in sorted((directory / "default").glob("seed_*/test_episodes.json")):
        episodes.extend(json.loads(path.read_text(encoding="utf-8")))
    components = ("gpu_idle_baseline_j", "gpu_prefill_increment_j", "gpu_decode_increment_j", "uplink_j", "downlink_j")
    energy = np.array([[r["energy_window"][k] for k in components] for r in episodes])
    fig, axes = plt.subplots(1, 2, figsize=(7.05, 2.7), width_ratios=[1.6, 1], layout="constrained")
    means = energy.mean(0)
    for ax, indices, labels, colors, title in (
            (axes[0], [0, 1, 2], ["Idle\nbaseline", "Prefill\nincrement", "Decode\nincrement"],
             ["#9BA4AC", "#497F9E", "#80A9B7"], "GPU"),
            (axes[1], [3, 4], ["Uplink", "Downlink"], ["#C48B5E", "#D8B18C"], "Radio · separate y scale")):
        bars = ax.bar(range(len(indices)), means[indices], color=colors)
        ax.bar_label(bars, labels=[f"{v:.3g}" for v in means[indices]], padding=3, fontsize=8)
        ax.set(xticks=range(len(indices)), xticklabels=labels, ylabel="Energy in observation window (J)")
        ax.margins(y=0.18)
        ax.set_title(title, fontsize=9)
    fig.suptitle("Illustrative phase-power model · not measured hardware energy", fontsize=9)
    save(fig, output / "energy")
    with (output / "energy_means.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["component", "mean_j", "evaluation_runs"])
        writer.writerows(zip(components, means, [len(episodes)] * len(components)))
