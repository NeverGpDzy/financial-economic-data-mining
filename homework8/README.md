# 作业8：统计套利之协整检验与配对交易模型

本目录保存作业8的完整可复现代码。教师提供的原始Word要求已复制到 `data/homework8/original/`，正文已原样提取为 `assignment.md`。

## 运行方法

```powershell
python -m homework8.main
```

如需重新从Baostock获取数据：

```powershell
python -m homework8.main --refresh
```

如需同步生成PPT：

```powershell
python -m homework8.main --build-ppt
```

## 口径说明

作业基础信息列出8个标的，但分步AI指令中只列出6个标的。为避免遗漏，本实现保留基础信息中的完整8标的池：7只股票加上证50指数，并遍历全部两两配对。

## 输出文件

核心结果输出到 `outputs/homework8/`：

- `cointegration_pairs.csv`：全部股票对EG两步法协整检验汇总。
- `aligned_close_prices.csv`：时间对齐后的收盘价矩阵。
- `best_pair_spread_zscore.csv`：最优配对价格、拟合值、残差、z-score和交易信号。
- `cointegration_pvalues_top12.png`：协整p值排序图。
- `best_pair_standardized_prices.png`：最优配对标准化价格对比图。
- `best_pair_spread.png`：残差价差时序图。
- `best_pair_zscore_signals.png`：z-score交易信号图。
- `homework8_report.md`：实验报告和深度问答题答案。
- `AI交互记录.md`、`AI代码审查与修复表.md`：AI过程与代码审查材料。
- `202331060205_丁致宇_作业8_协整套利投资.pptx`：课堂汇报PPT。

