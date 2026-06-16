# 实验一：金融非结构化数据预处理

本目录保存老师“实验一”资料对应的完整可复现实验代码。指导书原文已从 Word 抽取到 `assignment.md`，原始 Word 与数据包保存在 `data/experiment1/original/`，解压后的数据保存在 `data/experiment1/raw/`。

## 运行方式

在仓库根目录执行：

```powershell
python -m experiment1.main
```

## 完成内容

- 实验一：金融新闻清洗、交易日过滤、三分类情绪标注、周度情绪表、SQLite 入库。
- 实验二：按周构建 `H1t`、`H2t`、`H3t` 羊群效应指标。
- 实验三：对齐沪深300周收益，输出相关系数、格兰杰因果、最优滞后回归、残差 ADF 检验和图表。

## 主要输出

输出位于 `outputs/experiment1/`：

- `weekly_sentiment.csv`
- `weekly_herd_index.csv`
- `modeling_dataset.csv`
- `correlation_matrix.csv`
- `granger_results.csv`
- `regression_results.csv`
- `adf_results.csv`
- `experiment1.db`
- `*.png`
- `experiment1_report.md`

