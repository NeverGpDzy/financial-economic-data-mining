# 实验一：金融非结构化数据预处理与羊群效应预测分析报告

学生：丁致宇  
学号：202331060205

## 1. 作业要求理解

老师提供的《金融数据挖掘实验指导书 20260528》以“实验一”为文件夹名发布，但指导书正文实际包含三个连续环节：实验一生成周度情绪指标，实验二基于情绪指标构建羊群效应指数，实验三将羊群效应指数与沪深300周收益对齐并完成相关、格兰杰因果、滞后回归和残差 ADF 检验。三部分前后依赖，因此本次按完整链路完成。

关键口径如下：

- 新闻时间范围：2014-10-01 至 2015-10-31。
- 周度单位：按沪深300交易日历每 5 个交易日作为一周，先剔除非交易日新闻。
- 情绪输出：`week`、`WeekPositive`、`WeekNeutral`、`WeekNegative`、`P_t`。
- 羊群指标：`H1t = P_t - E(P_t)`，`H2t = 1 - |Positive-Negative|/(Positive+Negative)`，并结合“H2 越小羊群越强”的文字解释构造 `H3t = Norm(|H1t|) * (1 - Norm(H2t))`；同时在结果表保留 `H3t_formula_reference` 作为图片公式口径参考。
- 市场预测：沪深300日价格按同一 5 交易日分组计算周收益，自动检验 1-5 期格兰杰因果并选择最小 p 值滞后阶数做一次线性回归，再对残差做 ADF 检验。

## 2. 数据清洗与情绪标注

原始新闻共 60000 行。经过日期范围、正文非空、正文去重、交易日过滤后，进入周度统计的新闻为 30623 行，样本交易日范围为 2014-10-08 至 2015-10-21。

情绪标注采用 HuggingFace 开源中文金融情感模型 `yiyanghkust/finbert-tone-chinese`（BERT/Transformer 架构），本地批量推理实现三分类（正面/中性/负面）。该模型专门针对中文金融文本情绪识别训练，比通用主题分类模型更贴合指导书要求。标注方法为：bert。模型文件存放在仓库父目录 `models/` 下，不纳入 git 版本控制。

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
| H3t | -0.0175 |

格兰杰因果检验在 1-5 期中选择 p 值最小的滞后阶数，最优滞后为 5，对应 p 值为 0.1061。

最优滞后回归结果：

| 项目 | 数值 |
| --- | ---: |
| 回归方程 | `return_t = 0.022528 + -0.088783 * H3t_lag5` |
| R² | 0.0556 |
| beta p值 | 0.1146 |
| 样本数 | 46 |

残差 ADF 检验：

| 项目 | 数值 |
| --- | ---: |
| ADF统计量 | -4.8669 |
| p值 | 0.0000 |
| 5%水平残差平稳 | True |

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
