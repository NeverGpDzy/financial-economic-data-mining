# 作业6 AI代码审查与修复表

| 编号 | 严重性 | 审查发现 | 影响 | 修复动作 | 验证方式 | 状态 |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 高 | 题目写2006-2025，但教师数据实际只有2014-2024。 | 直接照题目年份会生成空样本或虚假结果。 | 按原始文件日期改为2014-2017训练、2018-2024样本外，并在报告说明口径差异。 | `data_audit.json` 记录实际日期范围。 | 已修复 |
| 2 | 高 | 未来1年FCFF标签会让2017样本使用2018信息。 | 如果用2017标签训练再回测2018，会产生样本外泄露。 | LGBM训练标签截至2016，只使用2014-2016带标签样本训练模型。 | `summary.json` 写明 `model_train_end_year=2016`。 | 已修复 |
| 3 | 中 | F4固资/总资产没有直接总资产字段。 | 因子无法按原定义直接计算。 | 用 `bps * total_share * assets_to_eqt` 估计总资产，保留公式说明。 | `annual_factor_panel.csv` 输出F4结果。 | 已修复 |
| 4 | 中 | 评分标准提到11个因子，但正文列F1-F10。 | 交付可能被认为少一个长期现金流因子。 | 在F10股息率后补充F11 FCFF收益率，并在报告解释该补充。 | `factor_quality_summary.csv` 包含F1-F11。 | 已修复 |
| 5 | 中 | 小样本LightGBM容易过拟合。 | 训练误差好但样本外分层失真。 | 限制 `max_depth=3`、`num_leaves=7`、`min_data_in_leaf=5` 并使用时间序列CV。 | `lgbm_cv_results.csv` 输出验证MSE。 | 已修复 |
| 6 | 中 | FCFF增速在FCFF接近0或由负转正时存在极端值。 | IC、OLS和LGBM会被少数异常标签主导。 | 新增 `target_fcff_growth_1y` 和 `target_fcff_growth_3y_ann`，按年度截面3σ缩尾后用于检验、训练和分层统计，原始标签保留。 | `annual_factor_panel.csv` 同时包含原始标签和缩尾标签。 | 已修复 |
| 7 | 高 | 年度面板未按当年年末上证50动态成分收口。 | 已退出或尚未进入上证50的股票会进入当年截面，扩大样本池。 | 按 `sz50_dynamic_components.csv` 保留财年12月31日仍在上证50且有年末行情的股票。 | `data_audit.json` 记录筛选前后行数，`annual_factor_panel.csv` 保留 `in_sz50_year_end`。 | 已修复 |
| 8 | 高 | 价格回测在年初使用上一财年年报得分，存在年报尚未披露的信息泄露。 | 2018年初无法知道2017年报完整财务指标。 | 价格回测使用两年滞后年度得分，例如2018持仓使用2016财年得分。 | `price_annual_returns.csv` 输出 `score_lag_years=2`，且 `hold_year-score_year=2`。 | 已修复 |
| 9 | 高 | 方案A原先按传统规则得分三等分，不是固定阈值硬筛。 | A组可能包含未通过关键阈值的股票，偏离作业要求。 | 方案A A组必须全部通过固定财务阈值；未通过股票按得分分入B/C对照组。 | `strategy_yearly_groups.csv` 输出 `traditional_hard_pass` 和 `grouping_method`。 | 已修复 |
