"""Executable pipeline for Experiment 3.

Run from repository root:

    python -m experiment3.main --build-pdf
"""

from __future__ import annotations

import argparse
import io
import json
import shutil
import sqlite3
import subprocess
import sys
from pathlib import Path

import pandas as pd

if sys.stdout.encoding and sys.stdout.encoding.lower().replace("-", "") != "utf8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

from . import config
from .analysis import (
    align_weekly_data,
    best_lag_from_importance,
    build_backward_features,
    build_features,
    build_hs300_weekly_return,
    build_natural_week_herd,
    build_quality_checks,
    compute_shap_values,
    contribution_summary,
    feature_importance_gain,
    load_herd_index,
    load_hs300_daily,
    residual_diagnostics,
    temporal_train_test_split,
    train_lgbm_model,
    valid_modeling_rows,
)
from .plots import generate_all_plots


def _save_df(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def _json_safe(obj):
    if hasattr(obj, "item"):
        return obj.item()
    if isinstance(obj, pd.Timestamp):
        return obj.strftime("%Y-%m-%d")
    if isinstance(obj, Path):
        return str(obj)
    return str(obj)


def _repo_rel(path: str | Path) -> str:
    p = Path(path)
    try:
        return p.relative_to(config.ROOT).as_posix()
    except ValueError:
        return p.as_posix()


def _markdown_table(df: pd.DataFrame, columns: list[str], max_rows: int | None = None) -> list[str]:
    view = df[columns].copy()
    if max_rows is not None:
        view = view.head(max_rows)
    headers = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows = [headers, sep]
    for _, row in view.iterrows():
        vals = []
        for col in columns:
            val = row[col]
            if isinstance(val, pd.Timestamp):
                vals.append(val.strftime("%Y-%m-%d"))
            elif isinstance(val, float):
                vals.append(f"{val:.6f}")
            else:
                vals.append(str(val))
        rows.append("| " + " | ".join(vals) + " |")
    return rows


def _metric_table_row(metrics: dict, direction: str) -> dict:
    return {
        "direction": direction,
        "MSE": metrics["mse"],
        "MAE": metrics["mae"],
        "RMSE": metrics["rmse"],
        "R2": metrics["r2"],
        "IC": metrics["ic"],
        "direction_acc": metrics["direction_acc"],
        "n_train": metrics["n_train"],
        "n_test": metrics["n_test"],
    }


def save_sqlite(tables: dict[str, pd.DataFrame]) -> None:
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(config.DB_FILE) as conn:
        for name, df in tables.items():
            df.to_sql(name, conn, if_exists="replace", index=False)


def write_ai_files(output_dir: Path) -> None:
    interaction = """# 实验三 AI交互记录

1. 读取老师最新版实验指导书，定位“实验三 基于LGBM的证券市场非线性预测与羊群效应因子分析”章节。
2. 检查仓库现有实验一、实验二代码和输出，确认实验三需要独立目录、独立输出和独立报告。
3. 读取实验二 `weekly_herd_index.csv`，确认可用字段为 `trade_date`、`P_t`、`H1t`、`H2t`、`H3t`。
4. 读取老师配套沪深300日度价格数据，按自然周取最后一个交易日收盘价并计算周收益率。
5. 将羊群效应指标与周收益率按自然周内连接对齐，剔除缺失值并生成建模数据集。
6. 构造 H3 滞后1到5期、3周/5周滚动均值和标准差、P_t历史特征、月份和季度虚拟变量，确保不使用同期 H3 预测同期收益。
7. 严格按时间顺序划分训练集和测试集，并只在训练窗口内部做简单参数选择。
8. 训练正向 LGBM 模型，输出测试集 MSE、MAE、RMSE、R²、IC、方向准确率、预测结果和残差诊断。
9. 计算 LightGBM Gain 特征重要性与 SHAP 值，生成 SHAP 特征重要性图和最优 H3 滞后因子的依赖图。
10. 构建反向模型，用滞后沪深300收益率预测当期 H3，对比情绪到收益、收益到情绪两个方向的传导强度。
11. 生成 CSV、SQLite、PNG 图表、Markdown 报告、PDF 报告、AI代码审查表和代码附录。
"""
    audit = """# 实验三 AI代码审查与修复表

| 编号 | 审查发现 | 影响 | 修复动作 | 验证 |
| --- | --- | --- | --- | --- |
| 1 | 实验一中已有部分实验三链路，但没有独立 `experiment3/`。 | 提交边界不清晰，老师难以按实验三单独复核。 | 新建 `experiment3/`、`data/experiment3/`、`outputs/experiment3/`，独立运行入口为 `python -m experiment3.main`。 | 输出文件全部写入 `outputs/experiment3/`。 |
| 2 | 实验三要求自然周对齐，而实验一、二基础周是5个交易日分组。 | 日期口径可能错位。 | 对 H3 和沪深300价格分别生成自然周标签，再按自然周内连接。 | `quality_checks.csv` 检查日期单调、关键字段无缺失。 |
| 3 | 若使用同期 H3 预测同期收益，会产生同周信息泄露。 | 样本外指标虚高。 | 正向模型只使用 `H3_lag1` 到 `H3_lag5` 与基于 `shift(1)` 的滚动统计特征。 | 质量检查项“未使用同期H3作为正向特征”为 True。 |
| 4 | 随机划分训练/测试会把未来样本混入训练。 | 时间序列预测失真。 | 按日期顺序前80%训练、后20%测试；参数选择只在训练窗口内部完成。 | `model_metrics.csv` 记录训练/测试样本数。 |
| 5 | 只给模型指标不足以满足实验报告要求。 | 缺少因子贡献、SHAP、残差和双向传导材料。 | 输出 Gain 重要性、SHAP 重要性、SHAP依赖图、残差图、双向对比表和SQLite数据库。 | `outputs/experiment3/` 包含对应 CSV 与 PNG。 |
"""
    (output_dir / "AI交互记录.md").write_text(interaction, encoding="utf-8")
    (output_dir / "AI代码审查与修复表.md").write_text(audit, encoding="utf-8")


def write_code_appendix(output_dir: Path) -> None:
    files = [
        config.PACKAGE_DIR / "config.py",
        config.PACKAGE_DIR / "analysis.py",
        config.PACKAGE_DIR / "plots.py",
        config.PACKAGE_DIR / "main.py",
    ]
    lines = ["# 实验三代码附录", ""]
    for path in files:
        rel = path.relative_to(config.ROOT).as_posix()
        lines += [f"## {rel}", "", "```python", path.read_text(encoding="utf-8").rstrip(), "```", ""]
    (output_dir / "实验三代码附录.md").write_text("\n".join(lines), encoding="utf-8")


def write_data_description(audit: dict, spec, feature_rows: int, train_rows: int, test_rows: int) -> None:
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    text = f"""# 实验三数据说明

## 输入数据

- `outputs/experiment2/weekly_herd_index.csv`：实验二生成的周度羊群效应指标，使用字段包括 `trade_date`、`P_t`、`H1t`、`H2t`、`H3t`。
- `data/experiment2/raw/沪深300日价格指数.xls`：老师提供的沪深300日度价格指数数据，实际格式为 GBK 制表符文本。

## 对齐口径

- 实验三按自然周对齐。程序先为羊群效应指标和沪深300价格分别生成 `natural_week`、`natural_week_start`、`natural_week_end`。
- 沪深300周收益率 `Ret_t` 使用自然周内最后一个交易日收盘价计算。
- 两个序列按自然周做内连接，剔除 `H3t`、`P_t`、`Ret_t` 缺失行。

## 样本摘要

- 实验二 H3 周样本：{audit['herd_rows']} 行。
- 沪深300自然周价格样本：{audit['hs300_weekly_rows']} 行。
- 自然周对齐后样本：{audit['aligned_rows']} 行。
- 对齐日期范围：{audit['aligned_start']} 至 {audit['aligned_end']}。
- 特征工程后可建模样本：{feature_rows} 行。
- 训练集样本：{train_rows} 行。
- 测试集样本：{test_rows} 行。

## 特征和标签

- 标签：`Ret_t`，当周沪深300收益率。
- H3滞后特征：{", ".join(spec.h3_cols)}。
- 情绪辅助特征：{", ".join(spec.sentiment_cols)}。
- 时间虚拟变量：月份和季度哑变量。
- 所有 H3 与 P_t 特征均通过 `shift(1)` 或更长滞后得到，不使用目标周同期情绪信息。

## 主要输出

- `outputs/experiment3/aligned_weekly_dataset.csv`：自然周对齐数据。
- `outputs/experiment3/feature_engineering.csv`：建模特征表。
- `outputs/experiment3/lgbm_forward_results.csv`：正向模型测试集预测结果。
- `outputs/experiment3/model_metrics.csv`：模型评估指标。
- `outputs/experiment3/feature_importance_gain.csv`、`outputs/experiment3/shap_importance.csv`：因子贡献表。
- `outputs/experiment3/experiment3.db`：SQLite 结果库。
"""
    (config.DATA_DIR / "数据说明.md").write_text(text, encoding="utf-8")


def write_markdown_report(summary: dict) -> str:
    config.REPORT_DIR.mkdir(parents=True, exist_ok=True)
    output = config.OUTPUT_DIR
    fwd = summary["forward_metrics"]
    bwd = summary["backward_metrics"]
    contrib = summary["contribution"]
    audit = summary["alignment_audit"]
    best_gain = summary["best_gain_lag"] or "N/A"
    best_shap = summary["best_shap_lag"] or summary["best_shap_feature"]
    stronger = summary["stronger_direction"]

    metrics_df = pd.DataFrame([
        _metric_table_row(fwd, "H3滞后特征 -> 沪深300周收益率"),
        _metric_table_row(bwd, "沪深300滞后收益率 -> H3"),
    ])
    gain_top = pd.DataFrame(summary["gain_top10"])
    shap_top = pd.DataFrame(summary["shap_top10"])
    comparison = pd.DataFrame(summary["bidirectional_comparison"])
    quality = pd.DataFrame(summary["quality_checks"])
    residual = pd.DataFrame(summary["residual_diagnostics"])

    lines = [
        "# 实验三：基于LGBM的证券市场非线性预测与羊群效应因子分析报告",
        "",
        f"学生：{config.STUDENT_NAME}  ",
        f"学号：{config.STUDENT_ID}",
        "",
        "## 1. 实验目标与实现口径",
        "",
        "本实验在实验二羊群效应指标基础上，将 H3 羊群效应指数与沪深300周收益率按自然周对齐，构造无未来信息泄露的时序机器学习数据集。正向模型使用过去1到5周的 H3 及历史滚动统计特征预测当周沪深300收益率；反向模型使用过去1到5周沪深300收益率预测当期 H3，用于检验市场收益对舆情羊群效应的反馈传导。",
        "",
        "关键约束包括：自然周内连接、按时间顺序切分训练/测试集、所有情绪特征均为滞后项、超参数选择只在训练窗口内部完成。",
        "",
        "## 2. 数据集说明",
        "",
        f"- 羊群效应输入：`{_repo_rel(config.INPUT_HERD_INDEX)}`。",
        f"- 沪深300价格输入：`{_repo_rel(config.INPUT_HS300_DAILY)}`。",
        f"- 自然周对齐样本：{audit['aligned_rows']} 行，日期范围 {audit['aligned_start']} 至 {audit['aligned_end']}。",
        f"- 特征工程后可建模样本：{summary['feature_rows']} 行；训练集 {fwd['n_train']} 行，测试集 {fwd['n_test']} 行。",
        f"- 标签：`Ret_t`，当周沪深300收益率。",
        f"- 正向特征：{', '.join(summary['feature_cols'])}。",
        "",
        "质量检查如下：",
        "",
    ]
    lines += _markdown_table(quality, ["check", "result", "detail"])
    lines += [
        "",
        "## 3. LGBM样本外预测结果",
        "",
        "测试集指标如下。R² 可能为负，表示模型在严格样本外条件下不如均值基准；本实验仍同时报告 IC 和方向准确率，以观察是否存在排序或方向信号。",
        "",
    ]
    lines += _markdown_table(metrics_df, ["direction", "MSE", "MAE", "RMSE", "R2", "IC", "direction_acc", "n_train", "n_test"])
    lines += [
        "",
        f"正向模型测试集 MSE={fwd['mse']:.6f}，MAE={fwd['mae']:.6f}，R²={fwd['r2']:.4f}，IC={fwd['ic']:.4f}，方向准确率={fwd['direction_acc']:.2%}。",
        "",
        "## 4. 因子贡献与最优滞后阶数",
        "",
        f"H3 类特征 Gain 总贡献占比为 {contrib['h3_gain_share']:.2%}，P_t 辅助情绪特征贡献占比为 {contrib['sentiment_gain_share']:.2%}，时间虚拟变量贡献占比为 {contrib['time_gain_share']:.2%}。",
        f"按 LightGBM Gain 排序，最优 H3 滞后特征为 `{best_gain}`；按 SHAP 平均绝对贡献排序，最优 H3 滞后特征为 `{best_shap}`。",
        "",
        "LightGBM Gain 特征重要性前10如下：",
        "",
    ]
    lines += _markdown_table(gain_top, ["feature", "gain", "split", "gain_share"])
    lines += [
        "",
        "SHAP 特征重要性前10如下：",
        "",
    ]
    lines += _markdown_table(shap_top, ["feature", "mean_abs_shap", "shap_share"])
    lines += [
        "",
        "## 5. SHAP非线性解释",
        "",
        f"SHAP依赖图使用 `{best_shap}` 作为横轴。若散点和拟合曲线呈现分段或弯曲形态，说明 H3 对收益预测的影响不是固定线性斜率，而是随情绪强度区间变化。这正是树模型和 SHAP 相比线性回归更适合本任务的原因：模型不预设线性关系，而是让数据决定阈值效应和非线性互动。",
        "",
        "## 6. 残差诊断",
        "",
    ]
    lines += _markdown_table(residual, ["metric", "value"])
    lines += [
        "",
        "残差图用于检查测试集误差是否集中在少数极端周。如果残差在个别周显著扩大，通常说明该周市场收益受到政策、流动性或突发事件影响，仅依赖舆情羊群指标难以完全解释。",
        "",
        "## 7. 双向传导与金融反身性",
        "",
    ]
    lines += _markdown_table(comparison, ["direction", "mse", "mae", "rmse", "r2", "ic", "direction_acc", "best_lag"])
    lines += [
        "",
        f"双向比较显示：{stronger}。从金融反身性角度看，舆情羊群效应可能通过投资者交易行为影响收益，而市场涨跌也会反过来改变新闻叙事和投资者情绪。若反向模型更强，说明“收益驱动舆情”的反馈链条在当前样本中更明显；若正向模型更强，则说明 H3 更接近可用的先行择时因子。",
        "",
        "## 8. 图表索引",
        "",
    ]
    for label, path in summary["plots"].items():
        lines.append(f"- `{label}`：`{_repo_rel(path)}`")
    lines += [
        "",
        "## 9. 输出文件",
        "",
        f"- `{_repo_rel(output / 'aligned_weekly_dataset.csv')}`：自然周对齐后的 H3、P_t 与沪深300收益率。",
        f"- `{_repo_rel(output / 'feature_engineering.csv')}`：建模特征表。",
        f"- `{_repo_rel(output / 'lgbm_forward_results.csv')}`：正向模型测试集预测结果。",
        f"- `{_repo_rel(output / 'model_metrics.csv')}`：正向和反向模型测试集指标。",
        f"- `{_repo_rel(output / 'feature_importance_gain.csv')}`、`{_repo_rel(output / 'shap_importance.csv')}`：特征贡献表。",
        f"- `{_repo_rel(output / 'experiment3.db')}`：SQLite结果库。",
        f"- `{_repo_rel(output / 'AI代码审查与修复表.md')}`、`{_repo_rel(output / 'AI交互记录.md')}`、`{_repo_rel(output / '实验三代码附录.md')}`：AI辅助与代码附录材料。",
        "",
        "## 10. 实验总结",
        "",
        "本实验按预测任务而不是同期解释任务组织特征，因此全部情绪特征均来自历史周。样本外结果显示，羊群效应与指数收益之间存在一定可检验的非线性预测关系，但强度受样本量和市场阶段影响明显。金融市场更适合非线性建模框架，因为情绪变量对收益的影响通常存在阈值、分段和反馈效应，固定线性系数难以稳定刻画这种关系。",
    ]
    text = "\n".join(lines)
    (config.OUTPUT_DIR / "experiment3_report.md").write_text(text, encoding="utf-8")
    (config.REPORT_DIR / "experiment3_report.md").write_text(text, encoding="utf-8")
    return text


def _latex_escape(text: str) -> str:
    return (
        str(text)
        .replace("\\", r"\textbackslash{}")
        .replace("&", r"\&")
        .replace("%", r"\%")
        .replace("$", r"\$")
        .replace("#", r"\#")
        .replace("_", r"\_")
        .replace("{", r"\{")
        .replace("}", r"\}")
        .replace("~", r"\textasciitilde{}")
        .replace("^", r"\textasciicircum{}")
    )


def _latex_tabular(df: pd.DataFrame, columns: list[str], aligns: str, max_rows: int | None = None) -> str:
    view = df[columns].copy()
    if max_rows is not None:
        view = view.head(max_rows)
    lines = [r"\begin{tabular}{" + aligns + "}", r"\toprule"]
    lines.append(" & ".join(_latex_escape(c) for c in columns) + r" \\")
    lines.append(r"\midrule")
    for _, row in view.iterrows():
        vals = []
        for col in columns:
            val = row[col]
            if isinstance(val, float):
                vals.append(f"{val:.6f}")
            else:
                vals.append(_latex_escape(val))
        lines.append(" & ".join(vals) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def write_latex_report(summary: dict) -> Path:
    latex_dir = config.REPORT_DIR / "experiment3_latex"
    latex_dir.mkdir(parents=True, exist_ok=True)
    metrics_df = pd.DataFrame([
        _metric_table_row(summary["forward_metrics"], "H3->收益"),
        _metric_table_row(summary["backward_metrics"], "收益->H3"),
    ])
    gain_top = pd.DataFrame(summary["gain_top10"]).head(8)
    shap_top = pd.DataFrame(summary["shap_top10"]).head(8)
    comparison = pd.DataFrame(summary["bidirectional_comparison"])
    quality = pd.DataFrame(summary["quality_checks"])
    residual = pd.DataFrame(summary["residual_diagnostics"])
    contrib = summary["contribution"]
    audit = summary["alignment_audit"]
    quality_table = _latex_tabular(quality, ["check", "result", "detail"], "p{0.28\\textwidth}cp{0.50\\textwidth}")
    metrics_table = _latex_tabular(metrics_df, ["direction", "MSE", "MAE", "RMSE", "R2", "IC", "direction_acc", "n_train", "n_test"], "lrrrrrrrr")
    gain_table = _latex_tabular(gain_top, ["feature", "gain", "split", "gain_share"], "lrrr")
    shap_table = _latex_tabular(shap_top, ["feature", "mean_abs_shap", "shap_share"], "lrr")
    residual_table = _latex_tabular(residual, ["metric", "value"], "lr")
    comparison_table = _latex_tabular(comparison, ["direction", "mse", "mae", "rmse", "r2", "ic", "direction_acc", "best_lag"], "lrrrrrrl")
    h3_gain_share = f"{contrib['h3_gain_share']:.2%}".replace("%", r"\%")
    sentiment_gain_share = f"{contrib['sentiment_gain_share']:.2%}".replace("%", r"\%")
    time_gain_share = f"{contrib['time_gain_share']:.2%}".replace("%", r"\%")
    best_gain_tex = _latex_escape(summary["best_gain_lag"] or "N/A")
    best_shap_tex = _latex_escape(summary["best_shap_lag"] or summary["best_shap_feature"])
    stronger_direction_tex = _latex_escape(summary["stronger_direction"])

    tex = rf"""\documentclass[UTF8,a4paper,12pt]{{ctexart}}
\usepackage{{geometry}}
\usepackage{{booktabs}}
\usepackage{{graphicx}}
\usepackage{{float}}
\usepackage{{array}}
\usepackage{{longtable}}
\usepackage{{hyperref}}
\usepackage{{xcolor}}
\geometry{{left=2.5cm,right=2.5cm,top=2.6cm,bottom=2.6cm}}
\graphicspath{{{{../../outputs/experiment3/}}}}
\hypersetup{{colorlinks=true,linkcolor=blue,urlcolor=blue}}
\setlength{{\parindent}}{{2em}}
\setlength{{\parskip}}{{0.35em}}

\title{{实验三：基于LGBM的证券市场非线性预测与羊群效应因子分析}}
\author{{{config.STUDENT_NAME}\\ 学号：{config.STUDENT_ID}}}
\date{{2026年6月}}

\begin{{document}}
\maketitle

\begin{{abstract}}
本报告基于实验二生成的周度羊群效应指数，将 H3 与沪深300自然周收益率对齐，构造滞后特征、滚动统计特征和时间虚拟变量，并使用 LightGBM 完成严格时序划分下的样本外预测。实验同时输出 Gain 特征重要性、SHAP 解释、残差诊断和双向传导模型，用于判断羊群效应是否具有非线性预测贡献，并分析收益与舆情之间的金融反身性。
\end{{abstract}}

\tableofcontents
\clearpage

\section{{实验设计}}
实验三的目标不是解释同期相关性，而是检验过去舆情羊群效应能否预测未来市场收益。程序将实验二 H3 指标和沪深300日度收盘价分别转换为自然周序列，再做内连接对齐。正向模型使用 H3 滞后1至5期、历史滚动均值和标准差、P\_t 历史特征以及月份、季度虚拟变量预测当周收益率，避免同期信息泄露。

\section{{数据集说明}}
自然周对齐样本为 {audit['aligned_rows']} 行，日期范围为 {audit['aligned_start']} 至 {audit['aligned_end']}。特征工程后可建模样本为 {summary['feature_rows']} 行，正向模型训练集 {summary['forward_metrics']['n_train']} 行，测试集 {summary['forward_metrics']['n_test']} 行。

\begin{{table}}[H]
\centering
\caption{{质量检查}}
\scriptsize
{quality_table}
\end{{table}}

\begin{{figure}}[H]
\centering
\includegraphics[width=0.95\textwidth]{{modeling_dataset_table.png}}
\caption{{实验三建模数据集截图}}
\end{{figure}}

\section{{样本外预测结果}}
\begin{{table}}[H]
\centering
\caption{{LGBM测试集预测指标}}
\scriptsize
{metrics_table}
\end{{table}}

正向模型测试集 MSE={summary['forward_metrics']['mse']:.6f}，MAE={summary['forward_metrics']['mae']:.6f}，R$^2$={summary['forward_metrics']['r2']:.4f}，IC={summary['forward_metrics']['ic']:.4f}。R$^2$ 若为负，说明严格样本外收益率预测仍弱于均值基准，但 IC 和方向准确率可以补充观察排序与方向信号。

\begin{{figure}}[H]
\centering
\includegraphics[width=0.95\textwidth]{{market_herd_timeseries.png}}
\caption{{自然周对齐：羊群效应与沪深300周收益率}}
\end{{figure}}

\begin{{figure}}[H]
\centering
\includegraphics[width=0.92\textwidth]{{prediction_vs_actual.png}}
\caption{{测试集实际收益率与LGBM预测值}}
\end{{figure}}

\section{{因子贡献与SHAP解释}}
H3 类特征 Gain 贡献占比为 {h3_gain_share}，P\_t 辅助情绪特征贡献占比为 {sentiment_gain_share}，时间虚拟变量贡献占比为 {time_gain_share}。按 Gain 排序，最优 H3 滞后特征为 \texttt{{{best_gain_tex}}}；按 SHAP 排序，最优 H3 滞后特征为 \texttt{{{best_shap_tex}}}。

\begin{{table}}[H]
\centering
\caption{{LightGBM Gain重要性前8}}
\scriptsize
{gain_table}
\end{{table}}

\begin{{table}}[H]
\centering
\caption{{SHAP重要性前8}}
\scriptsize
{shap_table}
\end{{table}}

\begin{{figure}}[H]
\centering
\includegraphics[width=0.88\textwidth]{{feature_importance_gain.png}}
\caption{{LightGBM Gain特征重要性}}
\end{{figure}}

\begin{{figure}}[H]
\centering
\includegraphics[width=0.88\textwidth]{{shap_feature_importance.png}}
\caption{{SHAP特征重要性}}
\end{{figure}}

\begin{{figure}}[H]
\centering
\includegraphics[width=0.78\textwidth]{{shap_dependence_best_lag.png}}
\caption{{最优H3滞后因子的SHAP依赖图}}
\end{{figure}}

\section{{残差诊断}}
\begin{{table}}[H]
\centering
\caption{{残差描述统计}}
\scriptsize
{residual_table}
\end{{table}}

\begin{{figure}}[H]
\centering
\includegraphics[width=0.92\textwidth]{{residual_timeseries.png}}
\caption{{测试集残差时序图}}
\end{{figure}}

\begin{{figure}}[H]
\centering
\includegraphics[width=0.70\textwidth]{{residual_distribution.png}}
\caption{{测试集残差分布}}
\end{{figure}}

\section{{双向传导与金融反身性}}
\begin{{table}}[H]
\centering
\caption{{双向建模结果对比}}
\scriptsize
{comparison_table}
\end{{table}}

\begin{{figure}}[H]
\centering
\includegraphics[width=0.92\textwidth]{{bidirectional_comparison.png}}
\caption{{双向建模样本外效果对比}}
\end{{figure}}

双向比较显示：{stronger_direction_tex}。这说明收益和情绪之间存在反馈循环：前期羊群情绪可能通过交易行为影响收益，而市场涨跌也会改变新闻叙事与投资者情绪。

\section{{结论}}
本实验完成了从实验二羊群效应指标到 LGBM 非线性预测、SHAP 解释、残差诊断和双向传导验证的完整链路。结果表明，羊群效应因子在严格样本外条件下的预测能力需要谨慎解释，但其滞后特征仍可通过 Gain 和 SHAP 量化贡献。相比线性回归，树模型更适合处理金融情绪变量的阈值效应、分段非线性和反馈关系。

\end{{document}}
"""
    tex_path = latex_dir / "main.tex"
    tex_path.write_text(tex, encoding="utf-8")
    return tex_path


def build_pdf(tex_path: Path) -> Path | None:
    if shutil.which("xelatex") is None:
        print("未找到 xelatex，跳过 PDF 编译。")
        return None
    latex_dir = tex_path.parent
    build_dir = latex_dir / "build"
    build_dir.mkdir(parents=True, exist_ok=True)
    cmd = ["xelatex", "-interaction=nonstopmode", "-halt-on-error", "-output-directory", str(build_dir), tex_path.name]
    for _ in range(2):
        result = subprocess.run(cmd, cwd=latex_dir, capture_output=True, text=True, encoding="utf-8", errors="replace")
        if result.returncode != 0:
            log_path = build_dir / "main.log"
            print(f"PDF 编译失败，日志：{log_path}")
            print(result.stdout[-1200:])
            return None
    pdf_src = build_dir / "main.pdf"
    if not pdf_src.exists():
        return None
    pdf_dst = config.OUTPUT_DIR / f"{config.STUDENT_ID}_{config.STUDENT_NAME}_实验三_LGBM非线性预测与羊群效应因子分析报告.pdf"
    shutil.copy2(pdf_src, pdf_dst)
    return pdf_dst


def run(build_pdf_report: bool = False) -> dict:
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    config.REPORT_DIR.mkdir(parents=True, exist_ok=True)

    print("实验三：读取实验二羊群效应指标和沪深300日度价格...")
    herd = load_herd_index()
    hs300_daily = load_hs300_daily()

    print("按自然周对齐 H3 与沪深300周收益率...")
    herd_weekly = build_natural_week_herd(herd)
    hs300_weekly = build_hs300_weekly_return(hs300_daily)
    aligned, audit = align_weekly_data(herd_weekly, hs300_weekly)

    print("构造无未来信息泄露的滞后与滚动特征...")
    featured, spec = build_features(aligned)
    featured_valid = valid_modeling_rows(featured, spec)
    quality = build_quality_checks(aligned, featured_valid, spec)
    train_df, test_df = temporal_train_test_split(featured_valid)

    print("训练正向 LGBM 模型：H3 滞后特征 -> 沪深300周收益率...")
    model, pred, forward_metrics, tuning = train_lgbm_model(train_df, test_df, spec)
    result = test_df[["natural_week", "natural_week_end", "price_week_close_date", "H3t", "P_t", "Ret_t"]].copy()
    result["actual_return"] = result["Ret_t"]
    result["predicted_return"] = pred
    result["residual"] = result["actual_return"] - result["predicted_return"]

    print("计算 Gain 重要性和 SHAP 可解释性...")
    importance = feature_importance_gain(model, spec.feature_cols)
    shap_values, shap_importance = compute_shap_values(model, test_df[spec.feature_cols])
    best_gain_lag = best_lag_from_importance(importance, "H3_lag")
    best_shap_lag = best_lag_from_importance(shap_importance, "H3_lag")
    best_shap_feature = best_shap_lag or (str(shap_importance.iloc[0]["feature"]) if not shap_importance.empty else spec.feature_cols[0])
    contrib = contribution_summary(importance, spec)
    residual_stats = residual_diagnostics(result)

    print("训练反向 LGBM 模型：沪深300滞后收益率 -> H3...")
    backward_featured, backward_spec = build_backward_features(aligned)
    backward_featured["H3_prev"] = backward_featured["H3t"].shift(1)
    backward_valid = valid_modeling_rows(backward_featured, backward_spec).dropna(subset=["H3_prev"]).reset_index(drop=True)
    backward_train, backward_test = temporal_train_test_split(backward_valid)
    backward_model, backward_pred, backward_metrics, backward_tuning = train_lgbm_model(
        backward_train,
        backward_test,
        backward_spec,
        direction_baseline=backward_test["H3_prev"].to_numpy(),
    )
    backward_importance = feature_importance_gain(backward_model, backward_spec.feature_cols)
    best_backward_lag = best_lag_from_importance(backward_importance, "Ret_lag")

    comparison = pd.DataFrame(
        [
            {
                "direction_short": "H3->收益",
                "direction": "H3滞后特征 -> 沪深300周收益率",
                **{k: forward_metrics[k] for k in ["mse", "mae", "rmse", "r2", "ic", "direction_acc", "n_train", "n_test"]},
                "best_lag": best_gain_lag,
            },
            {
                "direction_short": "收益->H3",
                "direction": "沪深300滞后收益率 -> H3",
                **{k: backward_metrics[k] for k in ["mse", "mae", "rmse", "r2", "ic", "direction_acc", "n_train", "n_test"]},
                "best_lag": best_backward_lag,
            },
        ]
    )
    if backward_metrics["r2"] > forward_metrics["r2"]:
        stronger_direction = f"反向模型 R²={backward_metrics['r2']:.4f} 高于正向模型 R²={forward_metrics['r2']:.4f}，收益对后续舆情羊群强度的反馈更强"
    elif forward_metrics["r2"] > backward_metrics["r2"]:
        stronger_direction = f"正向模型 R²={forward_metrics['r2']:.4f} 高于反向模型 R²={backward_metrics['r2']:.4f}，羊群效应对后续市场收益的预测更强"
    else:
        stronger_direction = f"正向与反向模型 R² 接近，正向 {forward_metrics['r2']:.4f}，反向 {backward_metrics['r2']:.4f}"

    print("保存 CSV、SQLite 和图表...")
    metrics_df = pd.DataFrame([
        _metric_table_row(forward_metrics, "H3滞后特征 -> 沪深300周收益率"),
        _metric_table_row(backward_metrics, "沪深300滞后收益率 -> H3"),
    ])
    _save_df(herd_weekly, config.OUTPUT_DIR / "natural_week_herd_index.csv")
    _save_df(hs300_weekly, config.OUTPUT_DIR / "hs300_natural_week_return.csv")
    _save_df(aligned, config.OUTPUT_DIR / "aligned_weekly_dataset.csv")
    _save_df(featured, config.OUTPUT_DIR / "feature_engineering.csv")
    _save_df(featured_valid, config.OUTPUT_DIR / "feature_engineering_modeling.csv")
    _save_df(result, config.OUTPUT_DIR / "lgbm_forward_results.csv")
    _save_df(metrics_df, config.OUTPUT_DIR / "model_metrics.csv")
    _save_df(tuning, config.OUTPUT_DIR / "forward_param_tuning.csv")
    _save_df(backward_tuning, config.OUTPUT_DIR / "backward_param_tuning.csv")
    _save_df(importance, config.OUTPUT_DIR / "feature_importance_gain.csv")
    _save_df(shap_importance, config.OUTPUT_DIR / "shap_importance.csv")
    _save_df(backward_importance, config.OUTPUT_DIR / "backward_feature_importance_gain.csv")
    _save_df(comparison, config.OUTPUT_DIR / "bidirectional_comparison.csv")
    _save_df(quality, config.OUTPUT_DIR / "quality_checks.csv")
    _save_df(residual_stats, config.OUTPUT_DIR / "residual_diagnostics.csv")

    save_sqlite(
        {
            "natural_week_herd_index": herd_weekly,
            "hs300_natural_week_return": hs300_weekly,
            "aligned_weekly_dataset": aligned,
            "feature_engineering": featured_valid,
            "lgbm_forward_results": result,
            "model_metrics": metrics_df,
            "feature_importance_gain": importance,
            "shap_importance": shap_importance,
            "backward_feature_importance_gain": backward_importance,
            "bidirectional_comparison": comparison,
            "quality_checks": quality,
            "residual_diagnostics": residual_stats,
        }
    )

    plots = generate_all_plots(
        aligned,
        featured_valid,
        result,
        importance,
        shap_importance,
        test_df[spec.feature_cols],
        shap_values,
        best_shap_feature,
        comparison,
        config.OUTPUT_DIR,
    )

    summary = {
        "student": {"name": config.STUDENT_NAME, "id": config.STUDENT_ID},
        "alignment_audit": audit,
        "feature_cols": spec.feature_cols,
        "feature_rows": int(len(featured_valid)),
        "quality_checks": quality.to_dict(orient="records"),
        "forward_metrics": forward_metrics,
        "backward_metrics": backward_metrics,
        "forward_best_params": forward_metrics.get("best_params", {}),
        "backward_best_params": backward_metrics.get("best_params", {}),
        "best_gain_lag": best_gain_lag,
        "best_shap_lag": best_shap_lag,
        "best_shap_feature": best_shap_feature,
        "best_backward_lag": best_backward_lag,
        "contribution": contrib,
        "gain_top10": importance.head(10).to_dict(orient="records"),
        "shap_top10": shap_importance.head(10).to_dict(orient="records"),
        "bidirectional_comparison": comparison.to_dict(orient="records"),
        "residual_diagnostics": residual_stats.to_dict(orient="records"),
        "stronger_direction": stronger_direction,
        "plots": plots,
    }
    (config.OUTPUT_DIR / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=_json_safe),
        encoding="utf-8",
    )

    print("生成报告、AI材料和代码附录...")
    write_ai_files(config.OUTPUT_DIR)
    write_code_appendix(config.OUTPUT_DIR)
    write_data_description(audit, spec, len(featured_valid), forward_metrics["n_train"], forward_metrics["n_test"])
    write_markdown_report(summary)
    tex_path = write_latex_report(summary)
    if build_pdf_report:
        pdf_path = build_pdf(tex_path)
        if pdf_path is not None:
            summary["pdf_report"] = str(pdf_path)
            (config.OUTPUT_DIR / "summary.json").write_text(
                json.dumps(summary, ensure_ascii=False, indent=2, default=_json_safe),
                encoding="utf-8",
            )
            print(f"PDF报告已生成：{pdf_path}")

    print(f"完成。输出目录：{config.OUTPUT_DIR}")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="实验三：基于LGBM的证券市场非线性预测与羊群效应因子分析")
    parser.add_argument("--build-pdf", action="store_true", help="同时编译 LaTeX 版 PDF 报告")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run(build_pdf_report=args.build_pdf)


if __name__ == "__main__":
    main()
