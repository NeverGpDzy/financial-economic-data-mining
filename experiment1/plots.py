"""Plot generation for experiment 1 outputs."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from . import config


def _setup_font() -> None:
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False


def plot_weekly_sentiment(weekly: pd.DataFrame) -> Path:
    _setup_font()
    path = config.OUTPUT_DIR / "weekly_sentiment_counts.png"
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(weekly["week"], weekly["WeekPositive"], label="正面", color="#2f9e44")
    ax.bar(weekly["week"], weekly["WeekNeutral"], bottom=weekly["WeekPositive"], label="中性", color="#868e96")
    ax.bar(
        weekly["week"],
        weekly["WeekNegative"],
        bottom=weekly["WeekPositive"] + weekly["WeekNeutral"],
        label="负面",
        color="#d9480f",
    )
    ax.set_title("周度新闻情绪数量")
    ax.set_xlabel("周末交易日")
    ax.set_ylabel("新闻数量")
    ax.legend(ncol=3)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_herd_index(herd: pd.DataFrame) -> Path:
    _setup_font()
    path = config.OUTPUT_DIR / "herd_index_timeseries.png"
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(herd["week"], herd["H1t"], label="H1 情绪偏离", color="#1971c2")
    ax.plot(herd["week"], herd["H2t"], label="H2 分歧度", color="#f08c00")
    ax.plot(herd["week"], herd["H3t"], label="H3 羊群强度", color="#c92a2a", linewidth=2)
    ax.set_title("周度羊群效应指标")
    ax.set_xlabel("周末交易日")
    ax.legend(ncol=3)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_market_relation(modeling: pd.DataFrame) -> Path:
    _setup_font()
    path = config.OUTPUT_DIR / "herd_vs_hs300_return.png"
    fig, ax1 = plt.subplots(figsize=(12, 5))
    ax2 = ax1.twinx()
    ax1.plot(modeling["week"], modeling["H3t"], color="#c92a2a", label="H3 羊群强度", linewidth=2)
    ax2.plot(modeling["week"], modeling["return"], color="#1864ab", label="沪深300周收益率", linewidth=1.8)
    ax1.set_title("羊群效应指标与沪深300周收益率")
    ax1.set_xlabel("周末交易日")
    ax1.set_ylabel("H3")
    ax2.set_ylabel("周收益率")
    lines = ax1.get_lines() + ax2.get_lines()
    ax1.legend(lines, [line.get_label() for line in lines], loc="upper left")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_regression_scatter(reg_dataset: pd.DataFrame, best_lag: int) -> Path:
    _setup_font()
    xcol = f"H3t_lag{best_lag}"
    path = config.OUTPUT_DIR / "lag_regression_scatter.png"
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(reg_dataset[xcol], reg_dataset["return"], color="#495057", alpha=0.8, label="样本点")
    ax.plot(reg_dataset[xcol], reg_dataset["fitted_return"], color="#c92a2a", label="线性拟合")
    ax.set_title(f"滞后 {best_lag} 期 H3 与沪深300周收益")
    ax.set_xlabel(xcol)
    ax.set_ylabel("return")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def generate_all_plots(weekly: pd.DataFrame, herd: pd.DataFrame, modeling: pd.DataFrame, reg_dataset: pd.DataFrame, best_lag: int) -> list[str]:
    paths = [
        plot_weekly_sentiment(weekly),
        plot_herd_index(herd),
        plot_market_relation(modeling),
        plot_regression_scatter(reg_dataset, best_lag),
    ]
    return [str(p.relative_to(config.ROOT)) for p in paths]

