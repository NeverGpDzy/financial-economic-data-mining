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
    build_features,
    build_herd_index,
    build_modeling_dataset,
    bidirectional_modeling,
    compute_shap,
    save_analysis_outputs,
    train_lgbm,
    temporal_train_test_split,
)
from .analysis import FEATURE_COLS
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


def write_report(summary: dict) -> None:
    config.REPORT_DIR.mkdir(parents=True, exist_ok=True)
    a = summary["audit"]
    fwd = summary["forward_metrics"]
    bwd = summary["backward_metrics"]
    bcomp = pd.DataFrame(summary["bidirectional_comparison"]) if summary["bidirectional_comparison"] else pd.DataFrame()
    best_feat = summary.get("best_shap_feature", "H3_lag1")

    # Forward best lag
    fwd_best = bcomp[bcomp["direction"].str.contains("forward")].iloc[0] if len(bcomp) > 0 else None
    bwd_best = bcomp[bcomp["direction"].str.contains("backward")].iloc[0] if len(bcomp) > 0 else None

    report = f"""# 实验一：金融非结构化数据预处理与羊群效应预测分析报告

学生：{config.STUDENT_NAME}
学号：{config.STUDENT_ID}

## 1. 作业要求理解

老师提供的《金融数据挖掘实验指导书 20260617》以"实验一"为文件夹名发布，但指导书正文实际包含三个连续环节：实验一生成周度情绪指标，实验二基于情绪指标构建羊群效应指数，实验三将羊群效应指数与沪深300周收益对齐并完成 LightGBM 非线性预测建模。三部分前后依赖，因此本次按完整链路完成。

关键口径如下：

- 新闻时间范围：{config.START_DATE} 至 {config.END_DATE}。
- 周度单位：按沪深300交易日历每 {config.TRADING_DAYS_PER_WEEK} 个交易日作为一周，先剔除非交易日新闻。
- 情绪标注：使用 `yiyanghkust/finbert-tone-chinese` BERT 模型本地 GPU 推理三分类。
- 情绪输出：`week`、`WeekPositive`、`WeekNeutral`、`WeekNegative`、`P_t`。
- 羊群指标：`H1t = P_t - E(P_t)`，`H2t = 1 - |Positive-Negative|/(Positive+Negative)`，`H3t = Norm(|H1t|) * (1 - Norm(H2t))`。
- 市场预测：采用 LightGBM 非线性时序建模，滞后 1~5 期特征 + 滚动统计 + 时间特征，前 80% 训练、后 20% 测试，辅以 SHAP 可解释性分析和双向传导验证。

## 2. 数据清洗与情绪标注

原始新闻共 {a['raw_news_rows']} 行。经过日期范围、正文非空、正文去重、交易日过滤后，进入周度统计的新闻为 {a['labeled_trading_day_rows']} 行，样本交易日范围为 {a['date_start']} 至 {a['date_end']}。

情绪标注采用 HuggingFace 开源中文金融情感模型 `yiyanghkust/finbert-tone-chinese`（BERT/Transformer 架构），本地 GPU 批量推理实现三分类（正面/中性/负面）。标注方法为：{a.get('labeling_method', 'bert')}。

周度情绪表已写入 `outputs/experiment1/weekly_sentiment.csv`，SQLite 表 `experiment1.db::weekly_sentiment`。

## 3. 羊群效应指标

`P_t` 表示正面新闻占正负情绪新闻的比例。`E(P_t)` 使用过去 4 个周度样本的滚动均值作为正常情绪基准。`H1t` 衡量本期情绪相对历史基准的偏离，`H2t` 衡量正负观点分歧程度，`H3t` 综合情绪异常程度与单边一致性。

羊群指标结果已写入 `outputs/experiment1/weekly_herd_index.csv`，图表 `outputs/experiment1/herd_index_timeseries.png`。

## 4. LightGBM 非线性预测建模

### 4.1 特征工程

对 H3t 构造滞后 1~5 期特征（`H3_lag1` ~ `H3_lag5`）、3 周和 5 周滚动均值/标准差、月份和季度时间特征。

### 4.2 时序划分

严格按时间顺序：前 80% 训练，后 20% 测试，杜绝数据泄露。

### 4.3 正向建模：H3 → 收益率

| 项目 | 数值 |
| --- | ---: |
| MSE | {fwd['mse']:.6f} |
| MAE | {fwd['mae']:.6f} |
| R² | {fwd['r2']:.4f} |
| 测试集样本数 | {fwd['n_test']} |

正向最优预测滞后（SHAP 特征重要性）：{fwd_best['best_lag'] if fwd_best is not None else 'N/A'}

### 4.4 SHAP 可解释性分析

SHAP 最重要特征：`{best_feat}`。图表见 `outputs/experiment1/shap_dependence.png`。

### 4.5 双向传导验证

| 方向 | MSE | MAE | R² | 最优滞后 |
| --- | ---: | ---: | ---: | ---: |
| H3→收益率 | {fwd['mse']:.6f} | {fwd['mae']:.6f} | {fwd['r2']:.4f} | {fwd_best['best_lag'] if fwd_best is not None else 'N/A'} |
| 收益率→H3 | {bwd['mse']:.6f} | {bwd['mae']:.6f} | {bwd['r2']:.4f} | {bwd_best['best_lag'] if bwd_best is not None else 'N/A'} |

### 4.6 金融反身性分析

{summary.get('reflexivity_notes', '正向模型（H3→收益率）和反向模型（收益率→H3）的对比揭示了舆情与市场之间的双向传导关系。')}

## 5. 输出文件索引

- `experiment1/assignment.md`：指导书正文按原内容抽取的 Markdown。
- `data/experiment1/original/`：老师发布的 Word 与 zip 原件。
- `data/experiment1/raw/`：解压后的新闻数据与沪深300价格数据。
- `data/experiment1/数据说明.md`：数据来源、格式和处理口径。
- `outputs/experiment1/weekly_sentiment.csv`：实验一周度情绪指标。
- `outputs/experiment1/weekly_herd_index.csv`：实验二羊群效应指标。
- `outputs/experiment1/modeling_dataset.csv`：建模对齐数据。
- `outputs/experiment1/feature_engineering.csv`：特征工程数据集。
- `outputs/experiment1/lgbm_forward_results.csv`：LightGBM 正向预测结果。
- `outputs/experiment1/bidirectional_comparison.csv`：双向建模对比。
- `outputs/experiment1/shap_importance.csv`：SHAP 特征重要性。
- `outputs/experiment1/lgbm_metrics.csv`：模型评估指标。
- `outputs/experiment1/experiment1.db`：SQLite 数据库存储结果。
- `outputs/experiment1/*.png`：周度情绪、羊群指标、收益对比、特征重要性、SHAP 依赖、残差、双向对比图。

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

    print("特征工程...")
    featured = build_features(modeling)
    feature_cols = [c for c in FEATURE_COLS if c in featured.columns]
    featured_valid = featured.dropna(subset=feature_cols + ["return"]).reset_index(drop=True)

    print("时序划分 + LightGBM 训练...")
    train_df, test_df = temporal_train_test_split(featured_valid)
    X_train = train_df[feature_cols]
    y_train = train_df["return"]
    X_test = test_df[feature_cols]
    y_test = test_df["return"]

    model, forward_metrics, y_pred = train_lgbm(X_train, y_train, X_test, y_test)
    print(f"  正向 R²={forward_metrics['r2']:.4f}, MSE={forward_metrics['mse']:.6f}")

    print("SHAP 可解释性分析...")
    shap_values, shap_importance = compute_shap(model, X_test)
    best_shap_feature = shap_importance.iloc[0]["feature"] if len(shap_importance) > 0 else "H3_lag1"

    print("双向传导验证...")
    _, _, bidir_comparison = bidirectional_modeling(modeling)
    backward_metrics = {}
    if len(bidir_comparison) > 0:
        bwd = bidir_comparison[bidir_comparison["direction"].str.contains("backward")]
        if len(bwd) > 0:
            backward_metrics = {
                "mse": float(bwd.iloc[0]["mse"]),
                "mae": float(bwd.iloc[0]["mae"]),
                "r2": float(bwd.iloc[0]["r2"]),
                "n_test": int(bwd.iloc[0]["n_test"]),
            }

    print("保存 CSV、SQLite 数据库和图表...")
    save_outputs(labeled, weekly_sentiment, weekly_herd, hs300_weekly, modeling)
    # Build test prediction df for plots and saving
    test_pred_df = test_df[["week", "H3t", "return"]].copy()
    test_pred_df["predicted_return"] = y_pred
    test_pred_df["residual"] = test_pred_df["return"] - test_pred_df["predicted_return"]

    save_analysis_outputs(
        featured, y_pred, bidir_comparison, shap_values, shap_importance,
        forward_metrics, backward_metrics, test_pred_df, modeling,
    )

    plots = generate_all_plots(
        weekly_sentiment, weekly_herd, modeling,
        shap_values, X_test, shap_importance,
        test_pred_df, bidir_comparison,
    )
    write_data_description(audit)

    summary = {
        "audit": audit,
        "weekly_rows": len(weekly_sentiment),
        "modeling_rows": len(modeling),
        "feature_rows": len(featured_valid),
        "forward_metrics": forward_metrics,
        "backward_metrics": backward_metrics,
        "bidirectional_comparison": bidir_comparison.to_dict(orient="records"),
        "best_shap_feature": best_shap_feature,
        "shap_top5": shap_importance.head(5).to_dict(orient="records"),
        "plots": plots,
        "reflexivity_notes": _build_reflexivity_notes(forward_metrics, backward_metrics, bidir_comparison),
    }
    (config.OUTPUT_DIR / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=_json_safe),
        encoding="utf-8",
    )
    write_report(summary)
    print(f"完成。输出目录：{config.OUTPUT_DIR}")
    return summary


def _build_reflexivity_notes(fwd, bwd, comp) -> str:
    if not fwd or not bwd:
        return ""
    fwd_r2 = fwd.get("r2", 0)
    bwd_r2 = bwd.get("r2", 0)
    if fwd_r2 > bwd_r2:
        return (
            f"正向模型 R²={fwd_r2:.4f} 高于反向 R²={bwd_r2:.4f}，"
            "说明羊群情绪对收益率的预测能力强于收益率对情绪的引导能力，"
            "舆情具有一定的先行指标价值。"
        )
    elif bwd_r2 > fwd_r2:
        return (
            f"反向模型 R²={bwd_r2:.4f} 高于正向 R²={fwd_r2:.4f}，"
            "说明市场收益率对羊群情绪的引导能力强于情绪对收益的预测，"
            "反映'收益驱动舆情'的反身性特征。"
        )
    else:
        return (
            f"双向模型 R²相近（正向 {fwd_r2:.4f}，反向 {bwd_r2:.4f}），"
            "情绪与收益之间存在对等的双向传导关系。"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="实验一：金融新闻情绪、羊群效应和LightGBM预测检验")
    return parser.parse_args()


def main() -> None:
    parse_args()
    run()


if __name__ == "__main__":
    main()
