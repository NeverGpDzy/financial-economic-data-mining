"""Plots for Experiment 2 outputs."""

from __future__ import annotations

from pathlib import Path
import textwrap

import matplotlib.pyplot as plt
import pandas as pd


def setup_matplotlib() -> None:
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.dpi"] = 130


def plot_herd_timeseries(herd: pd.DataFrame, output_path: Path) -> Path:
    setup_matplotlib()
    fig, ax = plt.subplots(figsize=(11, 5.8))
    x = pd.to_datetime(herd["trade_date"])
    ax.plot(x, herd["H1t_norm"], label="|H1| 标准化", linewidth=1.8)
    ax.plot(x, herd["H2t_strength"], label="一致性强度", linewidth=1.8)
    ax.plot(x, herd["H3t"], label="H3 羊群效应指数", linewidth=2.3, color="#b53d2a")
    ax.set_title("实验二：周度羊群效应指标时序")
    ax.set_xlabel("交易周末日期")
    ax.set_ylabel("标准化指标值")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_sentiment_vs_herd(herd: pd.DataFrame, output_path: Path) -> Path:
    setup_matplotlib()
    fig, ax1 = plt.subplots(figsize=(11, 5.8))
    x = pd.to_datetime(herd["trade_date"])
    ax1.plot(x, herd["P_t"], color="#2f6f9f", linewidth=1.8, label="P_t 周乐观占比")
    ax1.plot(x, herd["E_P_t"], color="#6f8f56", linewidth=1.5, linestyle="--", label="过去4周基准情绪")
    ax1.set_xlabel("交易周末日期")
    ax1.set_ylabel("情绪占比")
    ax1.set_ylim(0, 1)
    ax1.grid(alpha=0.25)

    ax2 = ax1.twinx()
    ax2.fill_between(x, herd["H3t"], color="#c47a2c", alpha=0.22, label="H3 羊群效应")
    ax2.plot(x, herd["H3t"], color="#b85c1d", linewidth=1.8)
    ax2.set_ylabel("H3")
    ax2.set_ylim(0, max(1.0, herd["H3t"].max() * 1.1))

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")
    ax1.set_title("情绪偏离与羊群效应对比")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_top_herd_weeks(top_weeks: pd.DataFrame, output_path: Path) -> Path:
    setup_matplotlib()
    fig, ax = plt.subplots(figsize=(10, 5.8))
    labels = pd.to_datetime(top_weeks["trade_date"]).dt.strftime("%Y-%m-%d")
    ax.barh(labels[::-1], top_weeks["H3t"].iloc[::-1], color="#8c4b3f")
    ax.set_title("H3 最高的10个交易周")
    ax.set_xlabel("H3 羊群效应指数")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_indicator_distribution(herd: pd.DataFrame, output_path: Path) -> Path:
    setup_matplotlib()
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2))
    specs = [
        ("H1t", "H1 情绪偏离度", "#496f8a"),
        ("H2t", "H2 分歧度", "#6f8f56"),
        ("H3t", "H3 羊群效应", "#b53d2a"),
    ]
    for ax, (col, title, color) in zip(axes, specs):
        ax.hist(herd[col], bins=14, color=color, alpha=0.84, edgecolor="white")
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_herd_table_snapshot(herd: pd.DataFrame, output_path: Path, rows: int = 12) -> Path:
    setup_matplotlib()
    cols = ["trade_date", "H1t", "H2t", "H2t_formula_raw", "H3t"]
    table_df = herd[cols].head(rows).copy()
    table_df["trade_date"] = pd.to_datetime(table_df["trade_date"]).dt.strftime("%Y-%m-%d")
    for col in cols[1:]:
        table_df[col] = table_df[col].map(lambda x: f"{float(x):.6f}")

    fig, ax = plt.subplots(figsize=(11.2, 5.8))
    ax.axis("off")
    table = ax.table(
        cellText=table_df.values,
        colLabels=["交易日", "H1t", "H2t(解释版)", "H2t(公式原文)", "H3t"],
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.45)
    ax.set_title("羊群效应指标时序表截图", pad=18)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_code_snippet(snippet: str, output_path: Path) -> Path:
    setup_matplotlib()
    wrapped = "\n".join(line[:118] for line in snippet.splitlines())
    fig, ax = plt.subplots(figsize=(12, 7.2))
    ax.axis("off")
    ax.set_title("实验二核心程序截图：H1/H2/H3计算", loc="left", pad=12)
    ax.text(
        0.01,
        0.97,
        wrapped,
        va="top",
        ha="left",
        family="Microsoft YaHei",
        fontsize=8.4,
        bbox={"boxstyle": "round,pad=0.5", "facecolor": "#f7f7f7", "edgecolor": "#cccccc"},
    )
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def extract_core_code_snippet(analysis_path: Path) -> str:
    lines = analysis_path.read_text(encoding="utf-8").splitlines()
    # Keep the formula block compact enough to be readable as a report screenshot.
    selected = lines[83:116]
    return "\n".join(textwrap.dedent(line) for line in selected)


def generate_all_plots(herd: pd.DataFrame, top_weeks: pd.DataFrame, output_dir: Path) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    analysis_path = Path(__file__).with_name("analysis.py")
    paths = {
        "herd_timeseries": plot_herd_timeseries(herd, output_dir / "herd_index_timeseries.png"),
        "sentiment_vs_herd": plot_sentiment_vs_herd(herd, output_dir / "sentiment_vs_herd.png"),
        "top_herd_weeks": plot_top_herd_weeks(top_weeks, output_dir / "top_herd_weeks.png"),
        "indicator_distribution": plot_indicator_distribution(herd, output_dir / "indicator_distribution.png"),
        "herd_table_snapshot": plot_herd_table_snapshot(herd, output_dir / "weekly_herd_index_table.png"),
        "core_code_snippet": plot_code_snippet(
            extract_core_code_snippet(analysis_path),
            output_dir / "core_program_snippet.png",
        ),
    }
    return {key: str(path) for key, path in paths.items()}
