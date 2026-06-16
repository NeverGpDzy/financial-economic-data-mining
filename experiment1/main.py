"""Executable pipeline for Experiment 1.

Run from repository root:

    python -m experiment1.main
"""

from __future__ import annotations

import argparse
import io
import json
import sys

import pandas as pd

if sys.stdout.encoding and sys.stdout.encoding.lower().replace("-", "") != "utf8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

from . import config
from .analysis import (
    build_herd_index,
    build_modeling_dataset,
    correlation_analysis,
    granger_analysis,
    lag_regression_and_adf,
    save_analysis_outputs,
)
from .data import (
    build_hs300_weekly,
    build_weekly_sentiment,
    clean_and_label_news,
    load_hs300_daily,
    load_news,
    make_trading_calendar,
    save_outputs,
    write_data_description,
)
from .plots import generate_all_plots


def _json_safe(obj):
    if hasattr(obj, "item"):
        return obj.item()
    if isinstance(obj, (pd.Timestamp,)):
        return str(obj)
    return str(obj)


def write_report(summary: dict, correlation: pd.DataFrame, regression: pd.DataFrame, adf: pd.DataFrame, granger: pd.DataFrame) -> None:
    config.REPORT_DIR.mkdir(parents=True, exist_ok=True)
    corr_h3 = float(correlation.loc["H3t", "return"])
    reg = regression.iloc[0]
    adf_row = adf.iloc[0]
    best_granger = granger.sort_values(["p_value", "lag"]).iloc[0]
    report = f"""# 实验一：金融非结构化数据预处理与羊群效应预测分析报告

学生：{config.STUDENT_NAME}  
学号：{config.STUDENT_ID}

## 1. 作业要求理解

老师提供的《金融数据挖掘实验指导书 20260528》以“实验一”为文件夹名发布，但指导书正文实际包含三个连续环节：实验一生成周度情绪指标，实验二基于情绪指标构建羊群效应指数，实验三将羊群效应指数与沪深300周收益对齐并完成相关、格兰杰因果、滞后回归和残差 ADF 检验。三部分前后依赖，因此本次按完整链路完成。

关键口径如下：

- 新闻时间范围：{config.START_DATE} 至 {config.END_DATE}。
- 周度单位：按沪深300交易日历每 {config.TRADING_DAYS_PER_WEEK} 个交易日作为一周，先剔除非交易日新闻。
- 情绪输出：`week`、`WeekPositive`、`WeekNeutral`、`WeekNegative`、`P_t`。
- 羊群指标：`H1t = P_t - E(P_t)`，`H2t = 1 - |Positive-Negative|/(Positive+Negative)`，并结合“H2 越小羊群越强”的文字解释构造 `H3t = Norm(|H1t|) * (1 - Norm(H2t))`；同时在结果表保留 `H3t_formula_reference` 作为图片公式口径参考。
- 市场预测：沪深300日价格按同一 5 交易日分组计算周收益，自动检验 1-5 期格兰杰因果并选择最小 p 值滞后阶数做一次线性回归，再对残差做 ADF 检验。

## 2. 数据清洗与情绪标注

原始新闻共 {summary['audit']['raw_news_rows']} 行。经过日期范围、正文非空、正文去重、交易日过滤后，进入周度统计的新闻为 {summary['audit']['labeled_trading_day_rows']} 行，样本交易日范围为 {summary['audit']['date_start']} 至 {summary['audit']['date_end']}。

本机未安装 `transformers`，且老师没有随数据包提供本地 BERT 模型文件。为了保证实验可复现，默认采用金融情绪词典打分器完成正面、中性、负面三分类；代码中将标题、分词正文和正文共同计分，标题及靠前词更高权重。该方案不依赖联网下载模型，适合提交时复跑。若后续本地已有 HuggingFace 模型缓存，可把 `experiment1/sentiment.py` 中的标注函数替换为指导书附录 BERT 推理。

周度情绪表已写入：

- `outputs/experiment1/weekly_sentiment.csv`
- SQLite 表：`experiment1.db::weekly_sentiment`

## 3. 羊群效应指标

`P_t` 表示正面新闻占正负情绪新闻的比例。`E(P_t)` 使用过去 4 个周度样本的滚动均值作为正常情绪基准。`H1t` 衡量本期情绪相对历史基准的偏离，`H2t` 衡量正负观点分歧程度，`H3t` 综合情绪异常程度与单边一致性。

羊群指标结果已写入：

- `outputs/experiment1/weekly_herd_index.csv`
- SQLite 表：`experiment1.db::weekly_herd_index`
- 图表：`outputs/experiment1/herd_index_timeseries.png`

## 4. 沪深300预测检验

同期 Pearson 相关系数：

| 指标 | 与沪深300周收益相关系数 |
| --- | ---: |
| H3t | {corr_h3:.4f} |

格兰杰因果检验在 1-5 期中选择 p 值最小的滞后阶数，最优滞后为 {int(best_granger['lag'])}，对应 p 值为 {best_granger['p_value']:.4f}。

最优滞后回归结果：

| 项目 | 数值 |
| --- | ---: |
| 回归方程 | `{reg['equation']}` |
| R² | {reg['r_squared']:.4f} |
| beta p值 | {reg['p_value_beta']:.4f} |
| 样本数 | {int(reg['n_obs'])} |

残差 ADF 检验：

| 项目 | 数值 |
| --- | ---: |
| ADF统计量 | {adf_row['adf_statistic']:.4f} |
| p值 | {adf_row['p_value']:.4f} |
| 5%水平残差平稳 | {adf_row['is_residual_stationary_5pct']} |

解释：若残差 ADF p 值小于 0.05，则该滞后回归残差平稳，可以认为不是典型伪回归；若大于等于 0.05，则需要谨慎解释预测关系。本次结果显示，羊群指标与沪深300周收益的线性相关性和预测强度应结合 p 值、R² 和残差平稳性共同判断。

## 5. 输出文件索引

- `experiment1/assignment.md`：指导书正文按原内容抽取的 Markdown。
- `data/experiment1/original/`：老师发布的 Word 与 zip 原件。
- `data/experiment1/raw/`：解压后的新闻数据与沪深300价格数据。
- `data/experiment1/数据说明.md`：数据来源、格式和处理口径。
- `outputs/experiment1/weekly_sentiment.csv`：实验一周度情绪指标。
- `outputs/experiment1/weekly_herd_index.csv`：实验二羊群效应指标。
- `outputs/experiment1/modeling_dataset.csv`：实验三对齐后的建模数据。
- `outputs/experiment1/correlation_matrix.csv`：相关系数矩阵。
- `outputs/experiment1/granger_results.csv`：1-5 期格兰杰检验结果。
- `outputs/experiment1/regression_results.csv`：最优滞后回归结果。
- `outputs/experiment1/adf_results.csv`：残差 ADF 检验结果。
- `outputs/experiment1/experiment1.db`：SQLite 数据库存储结果。
- `outputs/experiment1/*.png`：周度情绪、羊群指标、收益对比和回归散点图。

## 6. 运行方式

在仓库根目录执行：

```powershell
python -m experiment1.main
```
"""
    path = config.REPORT_DIR / "experiment1_report.md"
    path.write_text(report, encoding="utf-8")
    (config.OUTPUT_DIR / "experiment1_report.md").write_text(report, encoding="utf-8")


def run() -> dict:
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("读取老师配套数据...")
    hs300 = load_hs300_daily()
    calendar = make_trading_calendar(hs300)
    news = load_news()

    print("清洗新闻并标注三分类情绪...")
    labeled, audit = clean_and_label_news(news, calendar)
    weekly_sentiment = build_weekly_sentiment(labeled)

    print("构建羊群效应指标...")
    weekly_herd = build_herd_index(weekly_sentiment)
    hs300_weekly = build_hs300_weekly(hs300, calendar)
    modeling = build_modeling_dataset(weekly_herd, hs300_weekly)

    print("执行相关、格兰杰、滞后回归和残差ADF检验...")
    correlation = correlation_analysis(modeling)
    granger, best_lag = granger_analysis(modeling)
    regression, adf, reg_dataset = lag_regression_and_adf(modeling, best_lag)

    print("保存 CSV、SQLite 数据库和图表...")
    save_outputs(labeled, weekly_sentiment, weekly_herd, hs300_weekly, modeling)
    save_analysis_outputs(correlation, granger, regression, adf, reg_dataset)
    plots = generate_all_plots(weekly_sentiment, weekly_herd, modeling, reg_dataset, best_lag)
    write_data_description(audit)

    summary = {
        "audit": audit,
        "weekly_rows": len(weekly_sentiment),
        "modeling_rows": len(modeling),
        "best_lag": best_lag,
        "correlation_h3_return": float(correlation.loc["H3t", "return"]),
        "regression": regression.iloc[0].to_dict(),
        "adf": adf.iloc[0].to_dict(),
        "plots": plots,
    }
    (config.OUTPUT_DIR / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=_json_safe),
        encoding="utf-8",
    )
    write_report(summary, correlation, regression, adf, granger)
    print(f"完成。输出目录：{config.OUTPUT_DIR}")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="实验一：金融新闻情绪、羊群效应和沪深300预测检验")
    return parser.parse_args()


def main() -> None:
    parse_args()
    run()


if __name__ == "__main__":
    main()

