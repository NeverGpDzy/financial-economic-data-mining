"""Plotting utilities for Homework 3."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False


def plot_backtest_curve(
    result: pd.DataFrame, strategy_name: str, market_name: str, path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(result.index, result["cum_return"] * 100, label=strategy_name, linewidth=1.5)
    ax.plot(result.index, result["market_cum_return"] * 100, label=market_name, linewidth=1.5, alpha=0.7)
    ax.set_title("PEG策略回测：累计收益率对比", fontsize=14)
    ax.set_xlabel("日期")
    ax.set_ylabel("累计收益率 (%)")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_drawdown(result: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.fill_between(result.index, result["drawdown"] * 100, 0, alpha=0.4, color="red")
    ax.set_title("策略回撤", fontsize=14)
    ax.set_xlabel("日期")
    ax.set_ylabel("回撤 (%)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_model_comparison(
    comparison_df: pd.DataFrame, output_dir: Path,
) -> None:
    """Bar charts comparing CAPM vs two-factor model."""
    stocks = comparison_df["股票"].tolist()
    x = range(len(stocks))

    # Alpha comparison
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].bar([i - 0.15 for i in x], comparison_df["CAPM_α"] * 10000, width=0.3, label="CAPM α", alpha=0.8)
    axes[0].bar([i + 0.15 for i in x], comparison_df["二因子_α"] * 10000, width=0.3, label="二因子 α", alpha=0.8)
    axes[0].set_xticks(list(x))
    axes[0].set_xticklabels(stocks, rotation=30, fontsize=9)
    axes[0].set_title("Alpha 对比 (万分之一/日)", fontsize=12)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3, axis="y")

    # R² comparison
    axes[1].bar([i - 0.15 for i in x], comparison_df["CAPM_R²"], width=0.3, label="CAPM R-squared", alpha=0.8)
    axes[1].bar([i + 0.15 for i in x], comparison_df["二因子_R²"], width=0.3, label="Two-factor R-squared", alpha=0.8)
    axes[1].set_xticks(list(x))
    axes[1].set_xticklabels(stocks, rotation=30, fontsize=9)
    axes[1].set_title("R-squared 对比", fontsize=12)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3, axis="y")

    # Beta_PE
    colors = ["green" if v > 0 else "red" for v in comparison_df["二因子_β(pe)"]]
    axes[2].bar(list(x), comparison_df["二因子_β(pe)"], color=colors, alpha=0.8)
    axes[2].set_xticks(list(x))
    axes[2].set_xticklabels(stocks, rotation=30, fontsize=9)
    axes[2].set_title("PE_TTM 因子暴露 (beta2)", fontsize=12)
    axes[2].axhline(0, color="black", linewidth=0.5)
    axes[2].grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(output_dir / "model_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_pe_ttm_trends(
    stock_data: dict[str, pd.DataFrame], output_dir: Path,
) -> None:
    """Plot PE_TTM trends for all stocks."""
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()
    for i, (name, df) in enumerate(stock_data.items()):
        if "peTTM" in df.columns:
            axes[i].plot(df.index, df["peTTM"], linewidth=0.8)
            axes[i].set_title(name, fontsize=11)
            axes[i].set_ylabel("PE_TTM")
            axes[i].grid(True, alpha=0.3)
    fig.suptitle("各股票 PE_TTM 走势", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_dir / "pe_ttm_trends.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
