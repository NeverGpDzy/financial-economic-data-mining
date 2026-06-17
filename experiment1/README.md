# 实验一：金融非结构化数据预处理

本目录保存老师"实验一"资料对应的完整可复现实验代码。指导书原文已从 Word 抽取到 `assignment.md`，原始 Word 与数据包保存在 `data/experiment1/original/`，解压后的数据保存在 `data/experiment1/raw/`。

## 运行方式

在仓库根目录执行：

```powershell
python -m experiment1.main
```

## 完成内容

- 实验一：金融新闻清洗、交易日过滤、BERT 三分类情绪标注、周度情绪表、SQLite 入库。
- 实验二：按周构建 `H1t`、`H2t`、`H3t` 羊群效应指标。
- 实验三：对齐沪深300周收益，LightGBM 非线性时序建模、SHAP 可解释性分析、双向传导验证。

## 主要输出

输出位于 `outputs/experiment1/`：

- `weekly_sentiment.csv` — 周度情绪指标
- `weekly_herd_index.csv` — 羊群效应指标
- `modeling_dataset.csv` — 建模对齐数据
- `feature_engineering.csv` — 特征工程数据集
- `lgbm_forward_results.csv` — LightGBM 正向预测结果
- `bidirectional_comparison.csv` — 双向建模对比
- `shap_importance.csv` — SHAP 特征重要性
- `lgbm_metrics.csv` — 模型评估指标
- `experiment1.db` — SQLite 数据库
- `*.png` — 图表
- `experiment1_report.md` — 实验报告
