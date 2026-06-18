# 实验二 AI代码审查与修复表

| 编号 | 审查发现 | 影响 | 修复动作 | 验证 |
| --- | --- | --- | --- | --- |
| 1 | 实验二需要独立可交付，不能继续混放在实验一输出中。 | 提交材料边界不清晰。 | 新建 `experiment2/`、`data/experiment2/`、`outputs/experiment2/`。 | `python -m experiment2.main` 可独立运行。 |
| 2 | H2文字解释为“越小越一边倒”，若直接用正负差额会与正向羊群强度冲突。 | H3方向可能反了。 | 将 H2 定义为分歧度 `1 - abs(pos-neg)/(pos+neg)`，H3 使用 `1 - Norm(H2)`。 | `quality_checks.csv` 验证 H2、H3范围均为0~1。 |
| 3 | 4周基准若包含当周 P_t 会产生同周信息污染。 | H1偏离度被低估。 | 先 `shift(1)` 再做4周 rolling mean。 | 第一周因无历史基准被剔除，报告记录剔除数量。 |
| 4 | 输出只有CSV不满足报告要求。 | 缺少图表、数据库和AI审查材料。 | 同步输出 SQLite、PNG图表、Markdown摘要、LaTeX/PDF报告、AI交互记录、代码审查表和代码附录。 | `outputs/experiment2/` 与 `report/experiment2_latex/` 生成完整文件。 |
