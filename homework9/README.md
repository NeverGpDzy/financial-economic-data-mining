# 作业9：协整套利模型维护与风控对比实验

本目录保存作业九的完整可复现代码。教师发布的原始 Word 要求已复制到 `data/homework9/original/`，正文已原样提取到 `assignment.md`。

## 运行方法

```powershell
python -m homework9.main
```

如需重新从 Baostock 获取数据：

```powershell
python -m homework9.main --refresh
```

如需同步生成 PPT：

```powershell
python -m homework9.main --build-ppt
```

## 口径说明

- 原始父目录名为“作业九”，Word标题为“金融数据作业10/项目10”。本仓库按连续课程作业编号放在 `homework9/`。
- 2015-01-01至2018-01-01作为期初建模样本，2018年首个交易日至2024-12-31作为统一回测期。
- 配对交易收益采用等权多空总资金口径：单腿50%，即 `0.5 × (茅台收益 - 老窖收益)`。
- 方案B1严格按教师指令：每半年窗口内做ADF检验，检验通过才使用该窗口μ、σ和1.5σ阈值交易。
- 方案B2作为代码审查后的稳健性补充：每半年重置一次，用窗口开始日前最近三年数据做ADF检验和阈值估计，避免使用未来数据。

## 输出文件

核心结果输出到 `outputs/homework9/`：

- `strategy_metrics.csv`：方案A、方案B1、方案B2、茅台长期持有、老窖长期持有的绩效对比。
- `dynamic_windows.csv`：B1教师指令版半年ADF检验、动态中枢、阈值和风控空仓记录。
- `dynamic_causal_windows.csv`：B2防未来函数版半年ADF检验、动态中枢、阈值和风控空仓记录。
- `static_backtest.csv`、`dynamic_backtest.csv`、`dynamic_causal_backtest.csv`：逐日信号、持仓、收益和净值。
- `price_and_spread.png`、`static_thresholds.png`、`dynamic_thresholds.png`、`strategy_nav_comparison.png`、`drawdown_comparison.png`、`dynamic_adf_pvalues.png`：展示图表。
- `homework9_report.md`：实验报告。
- `AI交互记录.md`、`AI代码审查与修复表.md`：AI过程材料。
- `202331060205_丁致宇_作业9_模型维护与风控对比实验.pptx`：课堂展示PPT。
