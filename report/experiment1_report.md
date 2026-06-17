# 实验一：金融非结构化数据预处理与羊群效应预测分析报告

学生：丁致宇
学号：202331060205

## 1. 作业要求理解

老师提供的《金融数据挖掘实验指导书 20260617》以"实验一"为文件夹名发布，但指导书正文实际包含三个连续环节：实验一生成周度情绪指标，实验二基于情绪指标构建羊群效应指数，实验三将羊群效应指数与沪深300周收益对齐并完成 LightGBM 非线性预测建模。三部分前后依赖，因此本次按完整链路完成。

关键口径如下：

- 新闻时间范围：2014-10-01 至 2015-10-31。
- 周度单位：按沪深300交易日历每 5 个交易日作为一周，先剔除非交易日新闻。
- 情绪标注：使用 `yiyanghkust/finbert-tone-chinese` BERT 模型本地 GPU 推理三分类。
- 情绪输出：`week`、`WeekPositive`、`WeekNeutral`、`WeekNegative`、`P_t`。
- 羊群指标：`H1t = P_t - E(P_t)`，`H2t = 1 - |Positive-Negative|/(Positive+Negative)`，`H3t = Norm(|H1t|) * (1 - Norm(H2t))`。
- 市场预测：采用 LightGBM 非线性时序建模，滞后 1~5 期特征 + 滚动统计 + 时间特征，前 80% 训练、后 20% 测试，辅以 SHAP 可解释性分析和双向传导验证。

## 2. 数据清洗与情绪标注

原始新闻共 60000 行。经过日期范围、正文非空、正文去重、交易日过滤后，进入周度统计的新闻为 30623 行，样本交易日范围为 2014-10-08 至 2015-10-21。

情绪标注采用 HuggingFace 开源中文金融情感模型 `yiyanghkust/finbert-tone-chinese`（BERT/Transformer 架构），本地 GPU 批量推理实现三分类（正面/中性/负面）。标注方法为：bert。

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
| MSE | 0.005303 |
| MAE | 0.045029 |
| R² | -0.0783 |
| 测试集样本数 | 10 |

正向最优预测滞后（SHAP 特征重要性）：1

### 4.4 SHAP 可解释性分析

SHAP 最重要特征：`H3_lag1`。图表见 `outputs/experiment1/shap_dependence.png`。

### 4.5 双向传导验证

| 方向 | MSE | MAE | R² | 最优滞后 |
| --- | ---: | ---: | ---: | ---: |
| H3→收益率 | 0.005303 | 0.045029 | -0.0783 | 1 |
| 收益率→H3 | 0.058473 | 0.163948 | -0.0109 | 1 |

### 4.6 金融反身性分析

反向模型 R²=-0.0109 高于正向 R²=-0.0783，说明市场收益率对羊群情绪的引导能力强于情绪对收益的预测，反映'收益驱动舆情'的反身性特征。

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
