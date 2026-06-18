"""Plotting utilities for Homework 8."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from . import config


plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

COLORS = {
    "ink": "#1F2937",
    "slate": "#64748B",
    "blue": "#2563EB",
    "green": "#059669",
    "gold": "#D97706",
    "red": "#DC2626",
    "line": "#CBD5E1",
    "paper": "#F8FAFC",
}


def plot_pair_pvalues(pairs: pd.DataFrame, output: Path) -> Path:
    top = pairs.head(12).sort_values("p_value", ascending=True)
    labels = top["y_asset"].str.split("（").str[0] + "-" + top["x_asset"].str.split("（").str[0]
    fig, ax = plt.subplots(figsize=(11, 6.6), facecolor=COLORS["paper"])
    ax.set_facecolor("white")
    ax.barh(labels, top["p_value"], color=COLORS["blue"])
    ax.axvline(0.05, color=COLORS["red"], linestyle="--", linewidth=1.2, label="0.05阈值")
    ax.invert_yaxis()
    ax.set_xlabel("残差ADF p值")
    ax.set_title("协整检验显著性排序（p值越小越显著）", fontsize=15, fontweight="bold", color=COLORS["ink"])
    ax.grid(True, axis="x", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return output


def plot_best_pair_prices(best_detail: dict, pair_df: pd.DataFrame, output: Path) -> Path:
    y_label = best_detail["y_asset"]
    x_label = best_detail["x_asset"]
    y_norm = pair_df["y_price"] / pair_df["y_price"].iloc[0]
    x_norm = pair_df["x_price"] / pair_df["x_price"].iloc[0]

    fig, ax = plt.subplots(figsize=(11, 6.2), facecolor=COLORS["paper"])
    ax.set_facecolor("white")
    ax.plot(pair_df["date"], y_norm, label=y_label, color=COLORS["blue"], linewidth=1.8)
    ax.plot(pair_df["date"], x_norm, label=x_label, color=COLORS["gold"], linewidth=1.8)
    ax.set_title("最优协整配对标准化价格走势", fontsize=15, fontweight="bold", color=COLORS["ink"])
    ax.set_ylabel("标准化价格（首日=1）")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return output


def plot_spread(best_detail: dict, pair_df: pd.DataFrame, output: Path) -> Path:
    fig, ax = plt.subplots(figsize=(11, 5.8), facecolor=COLORS["paper"])
    ax.set_facecolor("white")
    mean = pair_df["spread"].mean()
    std = pair_df["spread"].std(ddof=1)
    ax.plot(pair_df["date"], pair_df["spread"], color=COLORS["green"], linewidth=1.4, label="残差价差")
    ax.axhline(mean, color=COLORS["ink"], linewidth=1, label="均值")
    ax.axhline(mean + 2 * std, color=COLORS["red"], linestyle="--", linewidth=1, label="+2σ")
    ax.axhline(mean - 2 * std, color=COLORS["red"], linestyle="--", linewidth=1, label="-2σ")
    ax.set_title(
        f"最优配对残差价差：ADF p值={best_detail['p_value']:.4f}",
        fontsize=15,
        fontweight="bold",
        color=COLORS["ink"],
    )
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return output


def plot_zscore(best_detail: dict, pair_df: pd.DataFrame, output: Path) -> Path:
    fig, ax = plt.subplots(figsize=(11, 5.8), facecolor=COLORS["paper"])
    ax.set_facecolor("white")
    ax.plot(pair_df["date"], pair_df["z_score"], color=COLORS["blue"], linewidth=1.3, label="z-score")
    ax.axhline(2, color=COLORS["red"], linestyle="--", linewidth=1, label="开仓阈值 +2")
    ax.axhline(-2, color=COLORS["green"], linestyle="--", linewidth=1, label="开仓阈值 -2")
    ax.axhline(0, color=COLORS["ink"], linewidth=1, label="平仓中轴")

    short_y = pair_df[pair_df["z_score"] > config.ZSCORE_ENTRY]
    long_y = pair_df[pair_df["z_score"] < -config.ZSCORE_ENTRY]
    exit_zone = pair_df[pair_df["z_score"].abs() <= 0.2]
    ax.scatter(short_y["date"], short_y["z_score"], color=COLORS["red"], s=32, label="做空Y/做多X")
    ax.scatter(long_y["date"], long_y["z_score"], color=COLORS["green"], s=32, label="做多Y/做空X")
    ax.scatter(exit_zone["date"], exit_zone["z_score"], color=COLORS["gold"], s=16, alpha=0.55, label="接近0平仓观察")

    ax.set_title("价差z-score与交易信号", fontsize=15, fontweight="bold", color=COLORS["ink"])
    ax.set_ylabel("z-score")
    ax.grid(True, alpha=0.25)
    ax.legend(ncol=2, fontsize=9)
    fig.tight_layout()
    fig.savefig(output, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return output

