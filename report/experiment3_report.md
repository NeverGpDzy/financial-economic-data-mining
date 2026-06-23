# 实验三：基于LGBM的证券市场非线性预测与羊群效应因子分析报告

学生：丁致宇  
学号：202331060205

## 1. 实验目标与实现口径

本实验在实验二羊群效应指标基础上，将 H3 羊群效应指数与沪深300周收益率按自然周对齐，构造无未来信息泄露的时序机器学习数据集。正向模型使用过去1到5周的 H3 及历史滚动统计特征预测当周沪深300收益率；反向模型使用过去1到5周沪深300收益率预测当期 H3，用于检验市场收益对舆情羊群效应的反馈传导。

关键约束包括：自然周内连接、按时间顺序切分训练/测试集、所有情绪特征均为滞后项、超参数选择只在训练窗口内部完成。

## 2. 数据集说明

- 羊群效应输入：`outputs/experiment2/weekly_herd_index.csv`。
- 沪深300价格输入：`data/experiment2/raw/沪深300日价格指数.xls`。
- 自然周对齐样本：50 行，日期范围 2014-10-20 至 2015-10-25。
- 特征工程后可建模样本：45 行；训练集 36 行，测试集 9 行。
- 标签：`Ret_t`，当周沪深300收益率。
- 正向特征：H3_lag1, H3_lag2, H3_lag3, H3_lag4, H3_lag5, H3_roll_mean_3, H3_roll_std_3, H3_roll_mean_5, H3_roll_std_5, P_t_lag1, P_t_lag2, P_t_lag3, P_t_roll_mean_3, month_1, month_2, month_3, month_4, month_5, month_6, month_7, month_8, month_9, month_10, month_11, month_12, quarter_1, quarter_2, quarter_3, quarter_4。

质量检查如下：

| check | result | detail |
| --- | --- | --- |
| 自然周对齐后无关键缺失 | True | H3、P_t、Ret_t 均完整。 |
| 自然周日期单调递增 | True | 所有样本按自然周结束日升序排列。 |
| 特征工程后无缺失 | True | 滞后与滚动特征首部缺失已剔除。 |
| 未使用同期H3作为正向特征 | True | 正向预测只使用 H3 的历史滞后和历史滚动统计。 |
| 样本外测试集存在 | True | 可建模样本 45 行，按时间顺序切分训练和测试。 |

## 3. LGBM样本外预测结果

测试集指标如下。R² 可能为负，表示模型在严格样本外条件下不如均值基准；本实验仍同时报告 IC 和方向准确率，以观察是否存在排序或方向信号。

| direction | MSE | MAE | RMSE | R2 | IC | direction_acc | n_train | n_test |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| H3滞后特征 -> 沪深300周收益率 | 0.003503 | 0.036376 | 0.059187 | -0.539814 | 0.152564 | 0.555556 | 36 | 9 |
| 沪深300滞后收益率 -> H3 | 0.059075 | 0.139555 | 0.243054 | 0.085004 | 0.733333 | 0.777778 | 36 | 9 |

正向模型测试集 MSE=0.003503，MAE=0.036376，R²=-0.5398，IC=0.1526，方向准确率=55.56%。

## 4. 因子贡献与最优滞后阶数

H3 类特征 Gain 总贡献占比为 40.84%，P_t 辅助情绪特征贡献占比为 57.71%，时间虚拟变量贡献占比为 1.45%。
按 LightGBM Gain 排序，最优 H3 滞后特征为 `H3_lag1`；按 SHAP 平均绝对贡献排序，最优 H3 滞后特征为 `H3_lag5`。

LightGBM Gain 特征重要性前10如下：

| feature | gain | split | gain_share |
| --- | --- | --- | --- |
| P_t_roll_mean_3 | 0.183076 | 15 | 0.281610 |
| H3_roll_std_3 | 0.126210 | 14 | 0.194138 |
| P_t_lag2 | 0.080444 | 27 | 0.123740 |
| P_t_lag3 | 0.072030 | 25 | 0.110798 |
| H3_lag1 | 0.062926 | 25 | 0.096793 |
| P_t_lag1 | 0.039636 | 21 | 0.060969 |
| H3_roll_std_5 | 0.034726 | 21 | 0.053416 |
| H3_lag2 | 0.014847 | 18 | 0.022837 |
| H3_lag5 | 0.009788 | 8 | 0.015055 |
| quarter_4 | 0.008300 | 15 | 0.012767 |

SHAP 特征重要性前10如下：

| feature | mean_abs_shap | shap_share |
| --- | --- | --- |
| H3_roll_std_3 | 0.010503 | 0.212484 |
| P_t_lag2 | 0.008234 | 0.166585 |
| H3_roll_std_5 | 0.006994 | 0.141497 |
| P_t_lag1 | 0.005554 | 0.112360 |
| P_t_lag3 | 0.003417 | 0.069139 |
| P_t_roll_mean_3 | 0.003105 | 0.062809 |
| H3_lag5 | 0.002752 | 0.055686 |
| quarter_4 | 0.002497 | 0.050511 |
| H3_lag1 | 0.001622 | 0.032818 |
| H3_lag3 | 0.001378 | 0.027876 |

## 5. SHAP非线性解释

SHAP依赖图使用 `H3_lag5` 作为横轴。若散点和拟合曲线呈现分段或弯曲形态，说明 H3 对收益预测的影响不是固定线性斜率，而是随情绪强度区间变化。这正是树模型和 SHAP 相比线性回归更适合本任务的原因：模型不预设线性关系，而是让数据决定阈值效应和非线性互动。

## 6. 残差诊断

| metric | value |
| --- | --- |
| residual_mean | -0.032930 |
| residual_std | 0.049180 |
| residual_min | -0.144697 |
| residual_max | 0.014041 |
| residual_abs_mean | 0.036376 |

残差图用于检查测试集误差是否集中在少数极端周。如果残差在个别周显著扩大，通常说明该周市场收益受到政策、流动性或突发事件影响，仅依赖舆情羊群指标难以完全解释。

## 7. 双向传导与金融反身性

| direction | mse | mae | rmse | r2 | ic | direction_acc | best_lag |
| --- | --- | --- | --- | --- | --- | --- | --- |
| H3滞后特征 -> 沪深300周收益率 | 0.003503 | 0.036376 | 0.059187 | -0.539814 | 0.152564 | 0.555556 | H3_lag1 |
| 沪深300滞后收益率 -> H3 | 0.059075 | 0.139555 | 0.243054 | 0.085004 | 0.733333 | 0.777778 | Ret_lag1 |

双向比较显示：反向模型 R²=0.0850 高于正向模型 R²=-0.5398，收益对后续舆情羊群强度的反馈更强。从金融反身性角度看，舆情羊群效应可能通过投资者交易行为影响收益，而市场涨跌也会反过来改变新闻叙事和投资者情绪。若反向模型更强，说明“收益驱动舆情”的反馈链条在当前样本中更明显；若正向模型更强，则说明 H3 更接近可用的先行择时因子。

## 8. 图表索引

- `market_herd_timeseries`：`outputs/experiment3/market_herd_timeseries.png`
- `prediction_vs_actual`：`outputs/experiment3/prediction_vs_actual.png`
- `residual_timeseries`：`outputs/experiment3/residual_timeseries.png`
- `residual_distribution`：`outputs/experiment3/residual_distribution.png`
- `feature_importance_gain`：`outputs/experiment3/feature_importance_gain.png`
- `shap_feature_importance`：`outputs/experiment3/shap_feature_importance.png`
- `shap_dependence`：`outputs/experiment3/shap_dependence_best_lag.png`
- `bidirectional_comparison`：`outputs/experiment3/bidirectional_comparison.png`
- `modeling_dataset_table`：`outputs/experiment3/modeling_dataset_table.png`
- `core_program_snippet`：`outputs/experiment3/core_program_snippet.png`

## 9. 输出文件

- `outputs/experiment3/aligned_weekly_dataset.csv`：自然周对齐后的 H3、P_t 与沪深300收益率。
- `outputs/experiment3/feature_engineering.csv`：建模特征表。
- `outputs/experiment3/lgbm_forward_results.csv`：正向模型测试集预测结果。
- `outputs/experiment3/model_metrics.csv`：正向和反向模型测试集指标。
- `outputs/experiment3/feature_importance_gain.csv`、`outputs/experiment3/shap_importance.csv`：特征贡献表。
- `outputs/experiment3/experiment3.db`：SQLite结果库。
- `outputs/experiment3/AI代码审查与修复表.md`、`outputs/experiment3/AI交互记录.md`、`outputs/experiment3/实验三代码附录.md`：AI辅助与代码附录材料。

## 10. 实验总结

本实验按预测任务而不是同期解释任务组织特征，因此全部情绪特征均来自历史周。样本外结果显示，羊群效应与指数收益之间存在一定可检验的非线性预测关系，但强度受样本量和市场阶段影响明显。金融市场更适合非线性建模框架，因为情绪变量对收益的影响通常存在阈值、分段和反馈效应，固定线性系数难以稳定刻画这种关系。