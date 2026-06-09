# 作业5：Alpha的持续性

本目录保存作业5的完整可复现代码。教师提供的原始 Word 要求和压缩包已复制到 `data/homework5/original/`，Word 正文已原样提取为 `assignment.md`。

## 运行方法

在仓库根目录执行：

```powershell
python -m homework5.main
```

如果需要重新联网获取沪深300指数：

```powershell
python -m homework5.main --refresh-market
```

如果要在分析完成后同步重建 PPT：

```powershell
python -m homework5.main --build-ppt
```

如果网络不可用，可以强制使用本地等权市场代理：

```powershell
python -m homework5.main --market-source proxy
```

## 作业口径

- 训练期：2019-01-01 至 2021-12-31。
- 主检验期：2022-01-01 至 2024-12-31，对应作业 AI 指令第4条。
- 稳健性检验期：2022-01-01 至 2023-12-31，对应“三、实验数据”的样本外说明。
- 市场基准：沪深300指数。
- 无风险利率：年化 1.5%，按 252 个交易日折算为日度。
- 排序对象：训练期 CAPM 年化 Alpha 最高的 Top20 股票。

## 输出文件

核心结果输出到 `outputs/homework5/`：

- `summary.json`：核心指标摘要。
- `alpha_train_2019_2021.csv`：训练期 CAPM 回归结果。
- `alpha_test_2022_2024.csv`：主检验期 CAPM 回归结果。
- `alpha_comparison_2022_2024.csv`：前后 Alpha、排名和变化对比。
- `top20_overlap_2022_2024.csv`：两期 Top20 重合股票。
- `group_persistence_2022_2024.csv`：按历史 Alpha 分组的持续性结果。
- `homework5_report.md`：简短分析报告。
- `AI辅助代码说明.md`：代码关键修改点。
- `AI代码审查与修复表.md`：AI辅助代码审查问题、影响、修复动作和验证记录。
- `*.png`：Alpha 对比、Top20 对比、重合度和分组持续性图表。
- `202331060205_丁致宇_作业5.pptx`：课堂展示 PPT。
- `202331060205_丁致宇_作业5.pdf`：由 PPT 导出的 PDF 备份。
