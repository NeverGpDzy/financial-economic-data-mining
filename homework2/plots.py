"""Plotting helpers for Homework 2."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def setup_chinese_font() -> None:
    plt.rcParams["font.sans-serif"] = [
        "Microsoft YaHei",
        "SimHei",
        "Noto Sans CJK SC",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False


def plot_backtest_curve(curve: pd.DataFrame, stock_name: str, save_path: Path) -> None:
    setup_chinese_font()
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.plot(curve.index, curve["stock_cum_return"], label=f"{stock_name} 持有收益", linewidth=2)
    ax.plot(curve.index, curve["market_cum_return"], label="沪深300 基准收益", linewidth=2)
    ax.axhline(0, color="#666666", linewidth=0.8, linestyle="--")
    ax.set_title("CAPM 高 Alpha 股票买入持有回测")
    ax.set_xlabel("日期")
    ax.set_ylabel("累计收益率")
    ax.yaxis.set_major_formatter(lambda x, _: f"{x:.0%}")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=160)
    plt.close(fig)


def plot_horizon_comparison(result: pd.DataFrame, stock_name: str, save_path: Path) -> None:
    setup_chinese_font()
    fig, ax = plt.subplots(figsize=(11, 5.5))
    x = range(len(result))
    width = 0.34
    labels = result["period"].tolist()

    ax.bar(
        [i - width / 2 for i in x],
        result["stock_total_return"],
        width=width,
        label=f"{stock_name}累计收益",
        color="#B43A3A",
    )
    ax.bar(
        [i + width / 2 for i in x],
        result["market_total_return"],
        width=width,
        label="沪深300累计收益",
        color="#1B7F5C",
    )
    ax.axhline(0, color="#666666", linewidth=0.8)
    ax.set_title("泸州老窖延长持有期回测对比")
    ax.set_ylabel("累计收益率")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=0)
    ax.yaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()

    for i, row in result.iterrows():
        for dx, value in [(-width / 2, row["stock_total_return"]), (width / 2, row["market_total_return"])]:
            va = "bottom" if value >= 0 else "top"
            offset = 0.015 if value >= 0 else -0.015
            ax.text(i + dx, value + offset, f"{value:.1%}", ha="center", va=va, fontsize=9)

    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=160)
    plt.close(fig)
