"""Plotting utilities for Homework 7."""

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
    "red": "#DC2626",
    "line": "#CBD5E1",
    "paper": "#F8FAFC",
}


def plot_stock_series(panel: pd.DataFrame, output_dir: Path = config.OUTPUT_DIR) -> dict[str, Path]:
    """Plot close and log-return time series for each stock."""
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}

    for (name, code), g in panel.groupby(["name", "display_code"], sort=False):
        fig, axes = plt.subplots(2, 1, figsize=(11, 7.2), sharex=True, facecolor=COLORS["paper"])
        for ax in axes:
            ax.set_facecolor("white")
            ax.grid(True, alpha=0.25)
            ax.spines[["top", "right"]].set_visible(False)

        axes[0].plot(g["date"], g["close"], color=COLORS["blue"], linewidth=1.8)
        axes[0].set_title(f"{name}（{code}）2024年收盘价", fontsize=14, fontweight="bold", color=COLORS["ink"])
        axes[0].set_ylabel("收盘价")

        axes[1].plot(g["date"], g["log_return"], color=COLORS["green"], linewidth=1.1)
        axes[1].axhline(0, color=COLORS["line"], linewidth=1)
        axes[1].set_title("对数收益率", fontsize=12, color=COLORS["ink"])
        axes[1].set_ylabel("log return")
        axes[1].set_xlabel("日期")

        fig.tight_layout()
        path = output_dir / f"{code}_{name}_price_return.png"
        fig.savefig(path, dpi=160, bbox_inches="tight")
        plt.close(fig)
        paths[f"{code}_{name}"] = path
    return paths


def plot_stationarity_summary(summary: pd.DataFrame, output: Path) -> Path:
    """Plot p-value comparison for price and log-return ADF tests."""
    fig, ax = plt.subplots(figsize=(11, 6), facecolor=COLORS["paper"])
    ax.set_facecolor("white")
    x = range(len(summary))
    width = 0.36

    ax.bar([i - width / 2 for i in x], summary["price_p_value"], width, label="收盘价ADF p值", color=COLORS["blue"])
    ax.bar([i + width / 2 for i in x], summary["return_p_value"], width, label="对数收益率ADF p值", color=COLORS["green"])
    ax.axhline(config.ADF_P_THRESHOLD, color=COLORS["red"], linestyle="--", linewidth=1.2, label="0.05阈值")
    ax.set_xticks(list(x))
    ax.set_xticklabels(summary["stock_name"], rotation=0)
    ax.set_ylabel("ADF p值")
    ax.set_title("收盘价与对数收益率ADF检验p值对比", fontsize=15, fontweight="bold", color=COLORS["ink"])
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(loc="upper right")
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return output

