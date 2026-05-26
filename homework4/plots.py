"""Visualization for Homework 4: factor analysis and backtest results."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def plot_nav_vs_market(result: pd.DataFrame, output_path: Path) -> None:
    """Plot strategy NAV vs market index NAV."""
    fig, ax = plt.subplots(figsize=(14, 6))

    ax.plot(result.index, result["nav"] / 1e4, label="策略净值", color="#1f77b4", linewidth=1.5)
    ax.plot(result.index, result["market_nav"] / 1e4, label="上证指数", color="#ff7f0e", linewidth=1.2, alpha=0.8)

    ax.set_title("多因子选股策略 vs 上证指数 (样本外回测 2024-2025)", fontsize=14, fontweight="bold")
    ax.set_xlabel("日期")
    ax.set_ylabel("净值 (万元)")
    ax.legend(fontsize=11)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f"))
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_drawdown(result: pd.DataFrame, output_path: Path) -> None:
    """Plot strategy drawdown."""
    peak = result["nav"].cummax()
    dd = (result["nav"] - peak) / peak

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.fill_between(result.index, 0, dd * 100, color="#d62728", alpha=0.3)
    ax.plot(result.index, dd * 100, color="#d62728", linewidth=1)

    ax.set_title("策略回撤曲线", fontsize=14, fontweight="bold")
    ax.set_xlabel("日期")
    ax.set_ylabel("回撤 (%)")
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
    ax.grid(True, alpha=0.3)
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_ic_series(ic_results: dict, output_path: Path) -> None:
    """Plot IC time series for each valid factor."""
    n = len(ic_results)
    if n == 0:
        return
    fig, axes = plt.subplots(n, 1, figsize=(14, 3 * n), squeeze=False)
    axes = axes[:, 0]

    for ax, (factor, info) in zip(axes, ic_results.items()):
        ic_series = info.get("ic_series", [])
        if ic_series:
            ax.plot(range(1, len(ic_series) + 1), ic_series, color="#2ca02c", linewidth=0.8)
            ax.axhline(y=info["IC_mean"], color="#d62728", linestyle="--", label=f'IC均值={info["IC_mean"]:.4f}')
            ax.axhline(y=0, color="gray", linestyle="-", alpha=0.5)
            ax.set_title(f"{factor} IC序列 ({info['grade']})", fontsize=12, fontweight="bold")
            ax.set_ylabel("IC")
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("月份序号")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_factor_weights(weights: dict, valid_factors: list[str], output_path: Path) -> None:
    """Bar chart of static factor weights."""
    fig, ax = plt.subplots(figsize=(10, 5))

    w_vals = [weights.get(f, 0) for f in valid_factors]
    colors = ["#1f77b4" if w > 0 else "#d62728" for w in w_vals]
    bars = ax.bar(valid_factors, w_vals, color=colors, edgecolor="white", linewidth=0.8)

    for bar, val in zip(bars, w_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{val:.4f}", ha="center", va="bottom" if val >= 0 else "top",
                fontsize=10, fontweight="bold")

    ax.set_title("多因子回归静态权重", fontsize=14, fontweight="bold")
    ax.set_ylabel("权重系数")
    ax.axhline(y=0, color="black", linewidth=0.5)
    ax.grid(True, alpha=0.2, axis="y")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_ic_ir_summary(ic_results: dict, output_path: Path) -> None:
    """IC/IR summary bar chart."""
    factors = list(ic_results.keys())
    ic_means = [ic_results[f]["IC_mean"] for f in factors]
    irs = [ic_results[f]["IR"] for f in factors]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    colors_ic = ["#2ca02c" if v > 0.02 else "#d62728" for v in ic_means]
    ax1.bar(factors, ic_means, color=colors_ic, edgecolor="white")
    ax1.axhline(y=0.02, color="orange", linestyle="--", label="IC=0.02 (预测能力)")
    ax1.axhline(y=0.05, color="green", linestyle="--", label="IC=0.05 (优秀)")
    ax1.set_title("因子IC均值", fontsize=13, fontweight="bold")
    ax1.set_ylabel("IC")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.2, axis="y")

    colors_ir = ["#2ca02c" if abs(v) > 0.1 else "#d62728" for v in irs]
    ax2.bar(factors, irs, color=colors_ir, edgecolor="white")
    ax2.axhline(y=0.1, color="green", linestyle="--", label="IR=0.1 (阈值)")
    ax2.axhline(y=-0.1, color="green", linestyle="--")
    ax2.set_title("因子IR (信息比率)", fontsize=13, fontweight="bold")
    ax2.set_ylabel("IR")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.2, axis="y")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_metrics_table(metrics: dict, output_path: Path) -> None:
    """Save backtest metrics as a styled table figure."""
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.axis("off")

    rows = [
        ["累计收益率", f"{metrics.get('累计收益率', 0):.2%}"],
        ["年化收益率", f"{metrics.get('年化收益率', 0):.2%}"],
        ["最大回撤", f"{metrics.get('最大回撤', 0):.2%}"],
        ["月度胜率", f"{metrics.get('月度胜率', 0):.2%}"],
        ["超额收益(vs上证指数)", f"{metrics.get('超额收益(vs上证指数)', 0):.2%}"],
        ["上证指数累计收益", f"{metrics.get('上证指数累计收益', 0):.2%}"],
        ["上证指数最大回撤", f"{metrics.get('上证指数最大回撤', 0):.2%}"],
        ["夏普比率", f"{metrics.get('夏普比率', 0):.2f}"],
        ["卡玛比率", f"{metrics.get('卡玛比率', 0):.2f}"],
        ["调仓次数", str(metrics.get("调仓次数", 0))],
    ]

    table = ax.table(
        cellText=rows,
        colLabels=["指标", "数值"],
        cellLoc="center",
        loc="center",
        colWidths=[0.5, 0.3],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.5)

    # Style header
    for i in range(2):
        table[(0, i)].set_facecolor("#2c3e50")
        table[(0, i)].set_text_props(color="white", fontweight="bold")

    # Alternate row colors
    for i in range(len(rows)):
        for j in range(2):
            if i % 2 == 0:
                table[(i + 1, j)].set_facecolor("#ecf0f1")

    ax.set_title("回测指标汇总", fontsize=14, fontweight="bold", pad=20)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
