# 作业6B：基于FCFF的价值投资

本目录保存作业6的完整可复现代码。教师提供的原始 Word 要求和压缩包已复制到 `data/homework6/original/`，Word 正文已原样提取为 `assignment.md`，数据包已解压到 `data/homework6/raw/`。

## 运行方法

在仓库根目录执行：

```powershell
python -m homework6.main
```

如果要在分析完成后同步重建 PPT：

```powershell
python -m homework6.main --build-ppt
```

## 作业口径

- 题目正文写数据范围为 2006-2025。
- 教师提供数据实际覆盖 2014-03-31 至 2024-12-31，年度财务截面为 2014-2024。
- 本次采用可复现口径：2014-2017 为训练研究窗口，2018-2024 为样本外股价回测窗口。
- LGBM 标签为未来 1 年 FCFF 增速，因此模型训练标签截至 2016，避免 2018 样本外回测泄露。
- 年度股票池按 `sz50_dynamic_components.csv` 收口，只保留对应财年 12 月 31 日仍在上证50且存在年末行情快照的股票。
- 年初价格回测不能使用尚未披露的上一财年年报，因此 2018 持仓使用 2016 财年得分，之后逐年滚动，`price_annual_returns.csv` 中 `hold_year - score_year = 2`。
- 方案A的 A 组是全部通过固定财务阈值的硬筛组合，B/C 组是未通过硬筛股票按传统规则得分形成的对照组。
- 未来 3 年 FCFF 分层回测因数据截至 2024，样本外可验证到 2021 年截面。
- FCFF 增速受接近 0 分母影响较大，检验、建模和 FCFF 分层统计使用年度截面 3σ 缩尾后的标签，原始标签保留在面板中备查。

## 输出文件

核心结果输出到 `outputs/homework6/`：

- `annual_factor_panel.csv`：年度因子面板。
- `factor_quality_summary.csv`：IC/IR、VIF、单因子 OLS 汇总。
- `feature_importance.csv`：LGBM 特征重要度。
- `strategy_yearly_groups.csv`：两套策略年度 A/B/C 分组和入选股票。
- `fcff_group_summary.csv`：FCFF 分层回测结果。
- `price_metrics.csv`：股价回测核心指标。
- `homework6_report.md`：实验报告和思考题作答。
- `AI交互记录.md`：分任务 AI 指令和人工修改记录。
- `AI代码审查与修复表.md`：代码审查、问题影响、修复动作和验证记录。
- `*.png`：因子质检、重要度、FCFF分层、净值、年度收益、财务画像图表。
- `202331060205_丁致宇_作业6.pptx`：课堂展示 PPT。
