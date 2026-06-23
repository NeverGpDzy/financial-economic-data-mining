"""Plotting utilities for Homework 9."""

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
    "purple": "#7C3AED",
    "line": "#CBD5E1",
    "paper": "#F8FAFC",
}


def _save(fig, output: Path) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return output


def plot_price_and_spread(panel: pd.DataFrame, output: Path) -> Path:
    df = panel.copy()
    df["date"] = pd.to_datetime(df["date"])
    fig, axes = plt.subplots(2, 1, figsize=(11.5, 7.2), sharex=True, facecolor=COLORS["paper"])
    for ax in axes:
        ax.set_facecolor("white")
        ax.grid(True, alpha=0.24)

    axes[0].plot(df["date"], df[config.MAOTAI] / df[config.MAOTAI].iloc[0], label="贵州茅台", color=COLORS["blue"])
    axes[0].plot(df["date"], df[config.LAOJIAO] / df[config.LAOJIAO].iloc[0], label="泸州老窖", color=COLORS["gold"])
    axes[0].axvline(pd.Timestamp(config.INITIAL_TRAIN_END), color=COLORS["red"], linestyle="--", linewidth=1)
    axes[0].set_title("标准化收盘价与期初建模分界", fontsize=15, fontweight="bold", color=COLORS["ink"])
    axes[0].set_ylabel("标准化价格")
    axes[0].legend()

    axes[1].plot(df["date"], df["spread"], color=COLORS["green"], linewidth=1.25)
    axes[1].axvline(pd.Timestamp(config.INITIAL_TRAIN_END), color=COLORS["red"], linestyle="--", linewidth=1, label="2018回测起点")
    axes[1].set_title("对数价差 spread = ln(茅台) - ln(老窖)", fontsize=13, color=COLORS["ink"])
    axes[1].set_ylabel("对数价差")
    axes[1].legend()
    fig.tight_layout()
    return _save(fig, output)


def plot_static_thresholds(static_bt: pd.DataFrame, output: Path) -> Path:
    df = static_bt.copy()
    df["date"] = pd.to_datetime(df["date"])
    fig, ax = plt.subplots(figsize=(11.5, 6.0), facecolor=COLORS["paper"])
    ax.set_facecolor("white")
    ax.plot(df["date"], df["spread"], color=COLORS["blue"], linewidth=1.2, label="对数价差")
    ax.plot(df["date"], df["mu"], color=COLORS["ink"], linewidth=1.0, label="静态中枢 μ")
    ax.plot(df["date"], df["upper"], color=COLORS["red"], linestyle="--", linewidth=1.0, label="+1.5σ")
    ax.plot(df["date"], df["lower"], color=COLORS["green"], linestyle="--", linewidth=1.0, label="-1.5σ")
    long_spread = df[df["target_position"] == 1]
    short_spread = df[df["target_position"] == -1]
    ax.scatter(long_spread["date"], long_spread["spread"], color=COLORS["green"], s=14, alpha=0.55, label="做多价差")
    ax.scatter(short_spread["date"], short_spread["spread"], color=COLORS["red"], s=14, alpha=0.55, label="做空价差")
    ax.set_title("方案A：静态中枢价差阈值与交易状态", fontsize=15, fontweight="bold", color=COLORS["ink"])
    ax.set_ylabel("对数价差")
    ax.grid(True, alpha=0.24)
    ax.legend(ncol=3, fontsize=9)
    fig.tight_layout()
    return _save(fig, output)


def plot_dynamic_thresholds(dynamic_bt: pd.DataFrame, output: Path) -> Path:
    df = dynamic_bt.copy()
    df["date"] = pd.to_datetime(df["date"])
    mode = str(df["mode"].dropna().iloc[0]) if "mode" in df.columns and df["mode"].notna().any() else "动态风控"
    fig, ax = plt.subplots(figsize=(11.5, 6.0), facecolor=COLORS["paper"])
    ax.set_facecolor("white")
    ax.plot(df["date"], df["spread"], color=COLORS["blue"], linewidth=1.2, label="对数价差")
    ax.plot(df["date"], df["mu"], color=COLORS["ink"], linewidth=1.0, label="动态中枢 μ")
    ax.plot(df["date"], df["upper"], color=COLORS["red"], linestyle="--", linewidth=1.0, label="+1.5σ")
    ax.plot(df["date"], df["lower"], color=COLORS["green"], linestyle="--", linewidth=1.0, label="-1.5σ")

    fail = df[df["can_trade"] == False]
    if not fail.empty:
        for _, group in fail.groupby("window"):
            ax.axvspan(group["date"].iloc[0], group["date"].iloc[-1], color=COLORS["red"], alpha=0.08)

    ax.set_title(f"方案B：{mode}ADF风控与动态阈值", fontsize=15, fontweight="bold", color=COLORS["ink"])
    ax.set_ylabel("对数价差")
    ax.grid(True, alpha=0.24)
    ax.legend(ncol=3, fontsize=9)
    fig.tight_layout()
    return _save(fig, output)


def plot_nav_comparison(backtests: dict[str, pd.DataFrame], output: Path) -> Path:
    fig, ax = plt.subplots(figsize=(11.5, 6.2), facecolor=COLORS["paper"])
    ax.set_facecolor("white")
    colors = [COLORS["red"], COLORS["green"], COLORS["purple"], COLORS["blue"], COLORS["gold"]]
    for (label, df), color in zip(backtests.items(), colors):
        show = df.copy()
        show["date"] = pd.to_datetime(show["date"])
        ax.plot(show["date"], show["nav"], label=label, linewidth=1.8, color=color)
    ax.axhline(1, color=COLORS["line"], linewidth=1)
    ax.set_title("策略净值曲线对比", fontsize=15, fontweight="bold", color=COLORS["ink"])
    ax.set_ylabel("净值（2018首个交易日=1）")
    ax.grid(True, alpha=0.24)
    ax.legend()
    fig.tight_layout()
    return _save(fig, output)


def plot_drawdown_comparison(backtests: dict[str, pd.DataFrame], output: Path) -> Path:
    fig, ax = plt.subplots(figsize=(11.5, 5.8), facecolor=COLORS["paper"])
    ax.set_facecolor("white")
    colors = [COLORS["red"], COLORS["green"], COLORS["purple"], COLORS["blue"], COLORS["gold"]]
    for (label, df), color in zip(backtests.items(), colors):
        show = df.copy()
        show["date"] = pd.to_datetime(show["date"])
        dd = show["nav"] / show["nav"].cummax() - 1
        ax.plot(show["date"], dd, label=label, linewidth=1.45, color=color)
    ax.set_title("最大回撤路径对比", fontsize=15, fontweight="bold", color=COLORS["ink"])
    ax.set_ylabel("回撤")
    ax.grid(True, alpha=0.24)
    ax.legend()
    fig.tight_layout()
    return _save(fig, output)


def plot_window_pvalues(dynamic_windows: pd.DataFrame, output: Path) -> Path:
    df = dynamic_windows.copy()
    mode = str(df["mode"].dropna().iloc[0]) if "mode" in df.columns and df["mode"].notna().any() else "动态风控"
    fig, ax = plt.subplots(figsize=(11.5, 5.8), facecolor=COLORS["paper"])
    ax.set_facecolor("white")
    colors = [COLORS["green"] if v else COLORS["red"] for v in df["can_trade"]]
    ax.bar(df["trade_window"], df["p_value"], color=colors)
    ax.axhline(config.ADF_P_THRESHOLD, color=COLORS["ink"], linestyle="--", linewidth=1.1, label="0.05阈值")
    ax.set_title(f"方案B：{mode}每半年ADF p值", fontsize=15, fontweight="bold", color=COLORS["ink"])
    ax.set_ylabel("ADF p值")
    ax.tick_params(axis="x", rotation=55, labelsize=8)
    ax.grid(True, axis="y", alpha=0.24)
    ax.legend()
    fig.tight_layout()
    return _save(fig, output)
