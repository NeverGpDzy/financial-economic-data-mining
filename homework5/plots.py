"""Plotting utilities for Homework 5."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import config


plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


COLORS = {
    "ink": "#1F2937",
    "slate": "#64748B",
    "teal": "#0F766E",
    "mint": "#14B8A6",
    "gold": "#D97706",
    "red": "#DC2626",
    "paper": "#F8FAFC",
    "line": "#CBD5E1",
}


def _pct_axis(ax):
    ax.yaxis.set_major_formatter(lambda x, pos: f"{x:.0%}")


def plot_alpha_scatter(comparison: pd.DataFrame, summary: dict, output: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 7), facecolor=COLORS["paper"])
    ax.set_facecolor("white")

    normal = comparison[~comparison["is_train_top20"]]
    top = comparison[comparison["is_train_top20"]]
    ax.scatter(
        normal["alpha_annual_train"],
        normal["alpha_annual_test"],
        s=48,
        color=COLORS["slate"],
        alpha=0.55,
        label="其他股票",
    )
    ax.scatter(
        top["alpha_annual_train"],
        top["alpha_annual_test"],
        s=72,
        color=COLORS["gold"],
        edgecolor=COLORS["ink"],
        linewidth=0.5,
        label="前期Top20",
    )

    lim_min = min(comparison["alpha_annual_train"].min(), comparison["alpha_annual_test"].min())
    lim_max = max(comparison["alpha_annual_train"].max(), comparison["alpha_annual_test"].max())
    pad = (lim_max - lim_min) * 0.08
    ax.plot([lim_min - pad, lim_max + pad], [lim_min - pad, lim_max + pad], "--", color=COLORS["line"], label="45度线")
    ax.axhline(0, color=COLORS["line"], linewidth=0.9)
    ax.axvline(0, color=COLORS["line"], linewidth=0.9)

    ax.set_title(f"Alpha持续性散点图：训练期 vs {summary['test_label']}", fontsize=16, fontweight="bold", color=COLORS["ink"])
    ax.set_xlabel("2019-2021 年化Alpha")
    ax.set_ylabel(f"{summary['test_label']} 年化Alpha")
    ax.xaxis.set_major_formatter(lambda x, pos: f"{x:.0%}")
    ax.yaxis.set_major_formatter(lambda x, pos: f"{x:.0%}")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    ax.text(
        0.02,
        0.98,
        f"Spearman={summary['spearman_corr']:.3f}\n重合度={summary['overlap_ratio']:.1%}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=11,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": COLORS["line"]},
    )

    fig.tight_layout()
    fig.savefig(output, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_top20_comparison(train_top: pd.DataFrame, summary: dict, output: Path) -> None:
    top = train_top.sort_values("alpha_annual_train", ascending=True).copy()
    y = np.arange(len(top))
    fig, ax = plt.subplots(figsize=(11, 9), facecolor=COLORS["paper"])
    ax.set_facecolor("white")

    ax.barh(y - 0.18, top["alpha_annual_train"], height=0.36, color=COLORS["teal"], label="训练期Alpha")
    ax.barh(y + 0.18, top["alpha_annual_test"], height=0.36, color=COLORS["gold"], label=f"{summary['test_label']} Alpha")
    ax.axvline(0, color=COLORS["line"], linewidth=1.0)
    ax.set_yticks(y)
    ax.set_yticklabels(top["code"])
    ax.xaxis.set_major_formatter(lambda x, pos: f"{x:.0%}")
    ax.set_title(f"历史Top20股票Alpha前后对比：{summary['test_label']}", fontsize=16, fontweight="bold", color=COLORS["ink"])
    ax.set_xlabel("年化Alpha")
    ax.grid(True, axis="x", alpha=0.25)
    ax.legend(loc="lower right")

    fig.tight_layout()
    fig.savefig(output, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_overlap(summary: dict, output: Path) -> None:
    overlap = summary["overlap_count"]
    non_overlap = summary["top_n"] - overlap
    fig, ax = plt.subplots(figsize=(7.5, 6.2), facecolor=COLORS["paper"])
    wedges, _ = ax.pie(
        [overlap, non_overlap],
        startangle=90,
        colors=[COLORS["teal"], COLORS["line"]],
        wedgeprops={"width": 0.36, "edgecolor": "white"},
    )
    ax.text(0, 0.08, f"{summary['overlap_ratio']:.1%}", ha="center", va="center", fontsize=30, fontweight="bold", color=COLORS["ink"])
    ax.text(0, -0.22, f"{overlap}/{summary['top_n']} 重合", ha="center", va="center", fontsize=13, color=COLORS["slate"])
    ax.set_title(f"两期Top20重合度：{summary['test_label']}", fontsize=16, fontweight="bold", color=COLORS["ink"])
    ax.legend(wedges, ["仍在未来Top20", "跌出未来Top20"], loc="lower center", bbox_to_anchor=(0.5, -0.08), ncol=2)
    fig.tight_layout()
    fig.savefig(output, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_group_persistence(group_df: pd.DataFrame, summary: dict, output: Path) -> None:
    labels = group_df["history_group"].astype(str).tolist()
    x = np.arange(len(labels))
    width = 0.36
    fig, ax = plt.subplots(figsize=(10.5, 6.6), facecolor=COLORS["paper"])
    ax.set_facecolor("white")

    ax.bar(x - width / 2, group_df["train_alpha_mean"], width, label="训练期均值", color=COLORS["teal"])
    ax.bar(x + width / 2, group_df["future_alpha_mean"], width, label=f"{summary['test_label']}均值", color=COLORS["gold"])
    ax.axhline(0, color=COLORS["line"], linewidth=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.yaxis.set_major_formatter(lambda y, pos: f"{y:.0%}")
    ax.set_title(f"按历史Alpha分组后的未来Alpha表现：{summary['test_label']}", fontsize=16, fontweight="bold", color=COLORS["ink"])
    ax.set_ylabel("年化Alpha均值")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_all(period_results: dict, output_dir: Path = config.OUTPUT_DIR) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for label, result in period_results.items():
        suffix = label.replace("-", "_")
        paths[f"scatter_{suffix}"] = output_dir / f"alpha_scatter_{suffix}.png"
        paths[f"top20_{suffix}"] = output_dir / f"top20_alpha_compare_{suffix}.png"
        paths[f"overlap_{suffix}"] = output_dir / f"overlap_{suffix}.png"
        paths[f"group_{suffix}"] = output_dir / f"group_persistence_{suffix}.png"
        plot_alpha_scatter(result["comparison"], result["summary"], paths[f"scatter_{suffix}"])
        plot_top20_comparison(result["train_top"], result["summary"], paths[f"top20_{suffix}"])
        plot_overlap(result["summary"], paths[f"overlap_{suffix}"])
        plot_group_persistence(result["group_persistence"], result["summary"], paths[f"group_{suffix}"])
    return paths

