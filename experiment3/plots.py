"""Plot generation for Experiment 3."""

from __future__ import annotations

from pathlib import Path
import textwrap

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def setup_matplotlib() -> None:
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.dpi"] = 130


def plot_market_herd_timeseries(aligned: pd.DataFrame, output_path: Path) -> Path:
    setup_matplotlib()
    x = pd.to_datetime(aligned["price_week_close_date"])
    fig, ax1 = plt.subplots(figsize=(11, 5.8))
    ax1.plot(x, aligned["H3t"], color="#b53d2a", linewidth=2.0, label="H3 羊群效应")
    ax1.set_ylabel("H3")
    ax1.grid(alpha=0.25)
    ax2 = ax1.twinx()
    ax2.bar(x, aligned["Ret_t"], width=4, color="#2f6f9f", alpha=0.32, label="沪深300周收益率")
    ax2.axhline(0, color="#555555", linewidth=0.8)
    ax2.set_ylabel("周收益率")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")
    ax1.set_title("自然周对齐：羊群效应与沪深300周收益率")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_prediction_vs_actual(result: pd.DataFrame, output_path: Path) -> Path:
    setup_matplotlib()
    x = pd.to_datetime(result["price_week_close_date"])
    fig, ax = plt.subplots(figsize=(10.5, 5.4))
    ax.plot(x, result["actual_return"], marker="o", linewidth=1.8, label="实际周收益率", color="#2f6f9f")
    ax.plot(x, result["predicted_return"], marker="s", linewidth=1.8, label="预测周收益率", color="#b53d2a")
    ax.axhline(0, color="#555555", linewidth=0.8)
    ax.set_title("测试集：实际收益率与LGBM预测值")
    ax.set_ylabel("周收益率")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_residual_timeseries(result: pd.DataFrame, output_path: Path) -> Path:
    setup_matplotlib()
    x = pd.to_datetime(result["price_week_close_date"])
    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    ax.plot(x, result["residual"], color="#4f5d75", marker="o", linewidth=1.6)
    ax.axhline(0, color="#b53d2a", linestyle="--", linewidth=1)
    ax.set_title("测试集残差时序图")
    ax.set_ylabel("残差（实际 - 预测）")
    ax.grid(alpha=0.25)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_residual_distribution(result: pd.DataFrame, output_path: Path) -> Path:
    setup_matplotlib()
    fig, ax = plt.subplots(figsize=(7.8, 4.8))
    ax.hist(result["residual"], bins=8, color="#6f8f56", edgecolor="white", alpha=0.86)
    ax.axvline(0, color="#b53d2a", linestyle="--", linewidth=1)
    ax.set_title("测试集残差分布")
    ax.set_xlabel("残差")
    ax.set_ylabel("频数")
    ax.grid(axis="y", alpha=0.22)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_feature_importance(importance: pd.DataFrame, output_path: Path) -> Path:
    setup_matplotlib()
    top = importance.head(12).iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, 5.8))
    ax.barh(top["feature"], top["gain"], color="#2f6f9f")
    ax.set_title("LightGBM Gain 特征重要性（Top 12）")
    ax.set_xlabel("Gain")
    ax.grid(axis="x", alpha=0.22)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_shap_importance(shap_importance: pd.DataFrame, output_path: Path) -> Path:
    setup_matplotlib()
    top = shap_importance.head(12).iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, 5.8))
    ax.barh(top["feature"], top["mean_abs_shap"], color="#8c4b3f")
    ax.set_title("SHAP 特征重要性（平均绝对SHAP值）")
    ax.set_xlabel("mean(|SHAP|)")
    ax.grid(axis="x", alpha=0.22)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_shap_dependence(X_test: pd.DataFrame, shap_values: np.ndarray, feature: str, output_path: Path) -> Path:
    setup_matplotlib()
    if feature not in X_test.columns:
        feature = X_test.columns[0]
    idx = list(X_test.columns).index(feature)
    fig, ax = plt.subplots(figsize=(7.8, 5.2))
    ax.scatter(X_test[feature], shap_values[:, idx], color="#b53d2a", edgecolors="#333333", linewidths=0.35, alpha=0.78)
    ax.axhline(0, color="#555555", linestyle="--", linewidth=0.9)
    ax.set_title(f"SHAP依赖图：{feature}")
    ax.set_xlabel(feature)
    ax.set_ylabel("SHAP值")
    if X_test[feature].nunique() > 1:
        z = np.polyfit(X_test[feature], shap_values[:, idx], 2 if len(X_test) >= 6 else 1)
        xs = np.linspace(X_test[feature].min(), X_test[feature].max(), 120)
        ax.plot(xs, np.polyval(z, xs), color="#2f6f9f", linewidth=1.5, label="非线性拟合趋势")
        ax.legend()
    ax.grid(alpha=0.22)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_bidirectional_comparison(comparison: pd.DataFrame, output_path: Path) -> Path:
    setup_matplotlib()
    view = comparison.copy()
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.6))
    specs = [("r2", "R²"), ("ic", "IC"), ("direction_acc", "方向准确率")]
    colors = ["#2f6f9f", "#b53d2a"]
    for ax, (col, title) in zip(axes, specs):
        ax.bar(view["direction_short"], view[col], color=colors[: len(view)])
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.22)
        for i, val in enumerate(view[col]):
            ax.text(i, val, f"{val:.3f}", ha="center", va="bottom" if val >= 0 else "top", fontsize=9)
    fig.suptitle("双向建模样本外效果对比", y=1.02)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_dataset_table(featured_valid: pd.DataFrame, output_path: Path, rows: int = 10) -> Path:
    setup_matplotlib()
    cols = ["natural_week_end", "price_week_close_date", "H3t", "P_t", "Ret_t", "H3_lag1", "H3_lag2"]
    table_df = featured_valid[cols].head(rows).copy()
    for col in ["natural_week_end", "price_week_close_date"]:
        table_df[col] = pd.to_datetime(table_df[col]).dt.strftime("%Y-%m-%d")
    for col in ["H3t", "P_t", "Ret_t", "H3_lag1", "H3_lag2"]:
        table_df[col] = table_df[col].map(lambda x: f"{float(x):.6f}")
    fig, ax = plt.subplots(figsize=(11.5, 5.3))
    ax.axis("off")
    table = ax.table(
        cellText=table_df.values,
        colLabels=["自然周结束", "收盘交易日", "H3", "P_t", "Ret_t", "H3_lag1", "H3_lag2"],
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.4)
    table.scale(1, 1.45)
    ax.set_title("实验三建模数据集截图", pad=18)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_code_snippet(snippet: str, output_path: Path) -> Path:
    setup_matplotlib()
    wrapped = "\n".join(line[:120] for line in snippet.splitlines())
    fig, ax = plt.subplots(figsize=(12, 7.0))
    ax.axis("off")
    ax.set_title("实验三核心程序截图：滞后特征与时序切分", loc="left", pad=12)
    ax.text(
        0.01,
        0.97,
        wrapped,
        va="top",
        ha="left",
        family="Consolas",
        fontsize=8.0,
        bbox={"boxstyle": "round,pad=0.5", "facecolor": "#f7f7f7", "edgecolor": "#cccccc"},
    )
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def extract_core_code_snippet(analysis_path: Path) -> str:
    lines = analysis_path.read_text(encoding="utf-8").splitlines()
    selected = []
    capture = False
    for line in lines:
        if line.startswith("def build_features"):
            capture = True
        if capture:
            selected.append(line)
        if capture and line.startswith("def build_backward_features"):
            break
    return "\n".join(textwrap.dedent(line) for line in selected[:46])


def generate_all_plots(
    aligned: pd.DataFrame,
    featured_valid: pd.DataFrame,
    result: pd.DataFrame,
    importance: pd.DataFrame,
    shap_importance: pd.DataFrame,
    X_test: pd.DataFrame,
    shap_values: np.ndarray,
    best_shap_feature: str,
    comparison: pd.DataFrame,
    output_dir: Path,
) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    analysis_path = Path(__file__).with_name("analysis.py")
    paths = {
        "market_herd_timeseries": plot_market_herd_timeseries(aligned, output_dir / "market_herd_timeseries.png"),
        "prediction_vs_actual": plot_prediction_vs_actual(result, output_dir / "prediction_vs_actual.png"),
        "residual_timeseries": plot_residual_timeseries(result, output_dir / "residual_timeseries.png"),
        "residual_distribution": plot_residual_distribution(result, output_dir / "residual_distribution.png"),
        "feature_importance_gain": plot_feature_importance(importance, output_dir / "feature_importance_gain.png"),
        "shap_feature_importance": plot_shap_importance(shap_importance, output_dir / "shap_feature_importance.png"),
        "shap_dependence": plot_shap_dependence(X_test, shap_values, best_shap_feature, output_dir / "shap_dependence_best_lag.png"),
        "bidirectional_comparison": plot_bidirectional_comparison(comparison, output_dir / "bidirectional_comparison.png"),
        "modeling_dataset_table": plot_dataset_table(featured_valid, output_dir / "modeling_dataset_table.png"),
        "core_program_snippet": plot_code_snippet(extract_core_code_snippet(analysis_path), output_dir / "core_program_snippet.png"),
    }
    return {key: str(path) for key, path in paths.items()}

