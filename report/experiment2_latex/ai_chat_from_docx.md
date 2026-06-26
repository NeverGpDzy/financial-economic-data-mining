# 实验二 AI交互记录

1.  资料整理：读取老师最新版实验指导书，定位"实验二 金融非结构化数据情感分析"章节，并提取为 Markdown。
2.  数据归档：复制老师发布的 Word、zip 原件，将配套原始数据放入 `data/experiment2/raw/`。
3.  要求分析：确认实验二依赖实验一输出的周度情绪表，核心指标为 P_t、E(P_t)、H1、H2、H3。
4.  公式实现：按正面/(正面+负面)重算 P_t；用过去4周 P_t 均值作为基准；计算 H1 情绪偏离。
5.  H2方向处理：根据指导书"H2越小代表越一边倒"的解释，将 H2 实现为正负观点分歧度，并在 H3 中取反向强度。
6.  结果输出：生成 `weekly_herd_index.csv`、SQLite数据库、质量检查、描述统计、Top羊群周、指标表截图和核心程序截图。
7.  报告生成：生成 Markdown 摘要、LaTeX/PDF 标准实验报告、AI代码审查表和代码附录，保证可提交材料完整。

![](D:\A西南石油\金融与经济数据挖掘\Code\report\experiment2_latex\ai_chat_media/media/image1.png){width="6.0in" height="3.2618055555555556in"}![](D:\A西南石油\金融与经济数据挖掘\Code\report\experiment2_latex\ai_chat_media/media/image2.png){width="6.0in" height="3.2618055555555556in"}![](D:\A西南石油\金融与经济数据挖掘\Code\report\experiment2_latex\ai_chat_media/media/image3.png){width="6.0in" height="3.2618055555555556in"}![](D:\A西南石油\金融与经济数据挖掘\Code\report\experiment2_latex\ai_chat_media/media/image4.png){width="6.0in" height="3.2618055555555556in"}![](D:\A西南石油\金融与经济数据挖掘\Code\report\experiment2_latex\ai_chat_media/media/image5.png){width="6.0in" height="3.2618055555555556in"}
