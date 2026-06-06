# 作业4B：中频短线量化全流程

本目录保存作业 4B 的完整可复现代码。最终提交材料位于 `outputs/homework4b/`。

## 文件说明

| 文件 | 说明 |
| --- | --- |
| `assignment.md` | 作业要求 Markdown 版 |
| `config.py` | 路径、时间区间、因子、LGBM 和回测参数 |
| `data.py` | 本地 Parquet 加载、四因子计算、清洗、数据集拆分与保存 |
| `factors.py` | 因子标准化、IC/IR、VIF、单因子 OLS |
| `models.py` | LGBM 时间序列交叉验证、最终模型训练、特征重要性 |
| `backtest.py` | 日频截面打分、Top5 等权调仓回测、绩效指标 |
| `plots.py` | 净值、回撤、IC、重要性和指标表图表 |
| `main.py` | 一键运行入口 |

## 数据要求

教师提供的本地 Parquet 数据默认放在仓库上级目录：

```text
../作业四-B/作业4B 配套数据 parquet v1.1/BAKdata
```

如果目录不同，修改 `config.py` 中的 `PARQUET_ROOT`。

主程序会在本地生成清洗后的 CSV：

```text
data/homework4b/train_data.csv
data/homework4b/test_data.csv
data/homework4b/full_data.csv
```

这些 CSV 体积较大，默认不提交到 Git。`data/homework4b/数据说明.md` 记录字段和因子口径。

## 最终研究口径

- 股票池：沪深300成分股，本地数据实际可用 299 只。
- 训练集：2020-01-02 至 2023-12-29。
- 样本外回测集：2024-01-02 至 2025-12-31。
- 因子：`Reversal`、`Liquidity`、`MoneyFlow`、`Value`。
- 去极值：交易日截面 3σ 缩尾。
- 标准化：交易日截面 Z-Score。
- 标签：个股次日超额收益 `next_excess_ret`。
- 模型：LightGBM 回归，5 折扩展窗口时间序列交叉验证。
- 交易：每日收盘后 Top5 等权调仓，单票 20%，单边交易成本 0.1%。
- 末日处理：最后一个回测日只结算旧持仓，不新开仓。

MoneyFlow 说明：作业原定义需要个股级 `buy_amount/sell_amount`。本地数据仅有市场级 `moneyflow_hsgt`，因此最终使用 `5日滚动签名成交额净流入 / circ_mv` 作为个股级代理，其中签名成交额为 `amount/10 * sign(ret)`。

## 运行方法

在仓库根目录执行：

```powershell
python -m homework4b.main
```

主流程会依次完成：

1. 读取本地 Parquet 数据。
2. 计算四个中频短线因子。
3. 清洗数据并保存训练集、回测集和全量数据。
4. 仅在训练集上做 IC/IR、VIF 和 OLS。
5. 训练 LGBM 并保存模型和特征重要性。
6. 在 2024-2025 样本外区间打分、调仓和回测。
7. 生成图表与 `summary.json`。

## 重跑结果

下列结果来自严格四因子口径和按日截面缩尾后的重跑。

| 指标 | 结果 |
| --- | ---: |
| 回测区间 | 2024-01-02 ~ 2025-12-31 |
| 回测天数 | 485 |
| 累计收益率 | 15.37% |
| 年化收益率 | 7.71% |
| 夏普比率 | 0.3215 |
| 最大回撤 | -40.59% |
| 日胜率 | 44.95% |
| 超额收益(累计) | -21.35% |
| 超额收益(年化) | -9.93% |
| 信息比率 | -0.0656 |
| 最终净值 | 1,153,739 |

结果文件：

| 文件 | 说明 |
| --- | --- |
| `outputs/homework4b/summary.json` | 回测核心指标 |
| `outputs/homework4b/ic_ir_results.csv` | IC/IR 结果 |
| `outputs/homework4b/vif_results.csv` | VIF 共线性检验 |
| `outputs/homework4b/ols_results.csv` | 单因子 OLS |
| `outputs/homework4b/feature_importance.csv` | LGBM 特征重要性 |
| `outputs/homework4b/backtest_result.csv` | 每日净值与收益 |
| `outputs/homework4b/trade_log.csv` | 每日调仓日志 |
| `outputs/homework4b/nav_vs_market.png` | 策略净值 vs 沪深300 |
| `outputs/homework4b/drawdown.png` | 回撤曲线 |

## 提交材料

最终提交材料已生成在 `outputs/homework4b/`：

| 文件 | 说明 |
| --- | --- |
| `202331060205_丁致宇_作业4B.pptx` | 最终 PPT |
| `202331060205_丁致宇_作业4B.pdf` | PPT 导出的 PDF |
| `202331060205_丁致宇_AI代码审查与修复表.docx` | AI 代码审查表 |
| `202331060205_丁致宇_AI交互记录.docx` | AI 交互记录，含 4 张截图 |
| `AI代码审核记录.md` | 审查与修复摘要 |
| `AI交互记录.md` | AI 交互过程记录 |

## 审查要点

已修复的关键问题：

- 删除原 AI 初稿额外加入的 `Momentum`、`Volatility`、`Turnover`、`VolumeChange`，回到题目要求的四因子。
- 因子质检、OLS 和模型训练只使用 2020-2023 训练集。
- 3σ 缩尾改为交易日截面处理，避免样本外预处理污染。
- 首日建仓扣单边成本，最后交易日不做无意义调仓。
- 基准首日收益置 0，使策略和沪深300从同一收盘起点比较。
- PPT 和 Word 附件均读取或同步最新结果，旧版硬编码收益数字已移除。
