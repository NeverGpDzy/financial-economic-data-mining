# 实验二：金融非结构化数据情感分析

本目录保存实验二的独立可复现代码。最新版实验指导书已复制到 `data/experiment2/original/`，实验二要求已提取为 `assignment.md`。

## 运行方法

在仓库根目录执行：

```powershell
python -m experiment2.main
```

实验二依赖实验一输出的 `outputs/experiment1/weekly_sentiment.csv`。如果该文件不存在，请先运行：

```powershell
python -m experiment1.main
```

## 指标口径

- `P_t`：周乐观情绪占比，按 `WeekPositive / (WeekPositive + WeekNegative)` 计算。
- `E(P_t)`：过去 4 周 `P_t` 的滚动平均值，不包含当周。
- `H1t`：情绪偏离度，`P_t - E(P_t)`。
- `H2t`：多空分歧度，`1 - abs(WeekPositive - WeekNegative) / (WeekPositive + WeekNegative)`；越小代表越一边倒。
- `H3t`：最终羊群效应指数，`Norm(abs(H1t)) * (1 - Norm(H2t))`。
- `H2t_formula_raw`、`H3t_formula_raw`：保留任务书公式原文转写口径，便于审查对照。

## 输出文件

核心结果输出到 `outputs/experiment2/`：

- `weekly_herd_index.csv`：实验二羊群效应指标表。
- `experiment2.db`：SQLite 数据库，包含输入、指标、质量检查、描述统计和 Top 羊群周。
- `herd_index_timeseries.png`：H1、H2一致性强度、H3时序图。
- `sentiment_vs_herd.png`：P_t、历史基准与H3对比图。
- `top_herd_weeks.png`：H3最高的10个交易周。
- `indicator_distribution.png`：H1、H2、H3分布图。
- `weekly_herd_index_table.png`：羊群效应指标时序表截图。
- `core_program_snippet.png`：系统主要程序截图。
- `experiment2_report.md`：实验摘要。
- `report/experiment2_latex/main.tex`：LaTeX版标准实验报告源码。
- `202331060205_丁致宇_实验二_羊群效应指数构建报告.pdf`：LaTeX编译后的可提交PDF报告。
- `AI交互记录.md`、`AI代码审查与修复表.md`：AI辅助与审查材料。
- `实验二代码附录.md`：本实验代码附录。
