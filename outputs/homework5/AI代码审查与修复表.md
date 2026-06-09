# 作业5 AI代码审查与修复表

| 编号 | 严重性 | 审查发现 | 影响 | 修复动作 | 验证方式 | 状态 |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 高 | `load_stock_prices()` 只读取 raw CSV，但 raw CSV 被 `.gitignore` 忽略。 | 新环境只有原始 zip 时，主流程会 `FileNotFoundError`，不满足代码可直接运行。 | 新增 `ensure_stock_price_csv()`，raw CSV 缺失时从 `data/homework5/original/上证50_日度行情_2019_2025.zip` 安全解压。 | 临时移走 raw CSV 后调用 `ensure_stock_price_csv()`，确认自动恢复 CSV 且文件大小正确。 | 已修复 |
| 2 | 高 | `.gitignore` 中作业五 PPTX 例外被后续全局 `*.pptx` 覆盖。 | 最终 PPT 不会被 Git 收录，提交材料可能缺失。 | 在全局 `*.pptx` 规则之后补充 `!outputs/homework5/**/*.pptx` 和 `!outputs/homework5/**/*.pdf`。 | `git status --untracked-files=all outputs/homework5` 已显示 PPTX/PDF 为可跟踪文件。 | 已修复 |
| 3 | 中 | 指定 `--market-source eastmoney/baostock` 时，只要缓存存在就直接读缓存，不校验缓存来源。 | 参数语义不可靠，可能用错市场基准来源或旧缓存。 | 新增缓存 meta 来源校验；显式指定来源时，仅当缓存来源匹配才复用，否则重新获取。 | 读取 meta 并用 `python -m homework5.main --market-source baostock` 验证缓存来源为 baostock 时可复用。 | 已修复 |
| 4 | 中 | 主流程不生成 PPT，需手动运行 `report/build_homework5_ppt.py`。 | 重跑分析后 PPT 可能保留旧结果。 | 新增 `--build-ppt` 参数，主流程结束后可同步重建 PPT。 | 运行 `python -m homework5.main --build-ppt`，确认分析和 PPT 均成功生成。 | 已修复 |
