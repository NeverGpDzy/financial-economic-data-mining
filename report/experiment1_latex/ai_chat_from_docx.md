# 实验一 AI交互记录

1.  需求拆解：读取《金融数据挖掘实验指导书》，确认实验一目标为金融新闻文本清洗、情感标注、按5个交易日生成周度情绪指标，并输出CSV与数据库。
2.  数据归档：将老师发布的实验指导书和配套数据归档到 `data/experiment1/original/`，将新闻数据与沪深300价格数据放入 `data/experiment1/raw/`。
3.  数据清洗：编写Python代码读取新闻表，筛选2014-10至2015-10样本，去除空文本、重复文本和非交易日新闻。
4.  情感标注：优先使用本地 `yiyanghkust/finbert-tone-chinese` 金融中文BERT模型进行正面/中性/负面三分类；若模型文件缺失，则使用透明金融词典规则作为可运行兜底。
5.  周度指标：按沪深300交易日历每5个交易日形成一周，统计 `WeekPositive`、`WeekNeutral`、`WeekNegative`、`NewsCount` 和 `P_t`。
6.  数据库存储：将周度情绪表、标注样本、羊群指标和建模数据写入SQLite数据库，并同步导出CSV。
7.  质量检查：检查样本日期、去重数量、交易日过滤结果、周度字段完整性和数据库输出。
8.  报告生成：输出Markdown报告、LaTeX/PDF报告、核心图表、数据库文件和代码附件，用于提交。

![](D:\A西南石油\金融与经济数据挖掘\Code\report\experiment1_latex\ai_chat_media/media/image1.png){width="6.0in" height="3.2618055555555556in"}![](D:\A西南石油\金融与经济数据挖掘\Code\report\experiment1_latex\ai_chat_media/media/image2.png){width="6.0in" height="3.2618055555555556in"}![](D:\A西南石油\金融与经济数据挖掘\Code\report\experiment1_latex\ai_chat_media/media/image3.png){width="6.0in" height="3.2618055555555556in"}![](D:\A西南石油\金融与经济数据挖掘\Code\report\experiment1_latex\ai_chat_media/media/image4.png){width="6.0in" height="3.2618055555555556in"}

![](D:\A西南石油\金融与经济数据挖掘\Code\report\experiment1_latex\ai_chat_media/media/image5.png){width="6.0in" height="3.2618055555555556in"}![](D:\A西南石油\金融与经济数据挖掘\Code\report\experiment1_latex\ai_chat_media/media/image6.png){width="6.0in" height="3.2618055555555556in"}![](D:\A西南石油\金融与经济数据挖掘\Code\report\experiment1_latex\ai_chat_media/media/image7.png){width="6.0in" height="3.2618055555555556in"}![](D:\A西南石油\金融与经济数据挖掘\Code\report\experiment1_latex\ai_chat_media/media/image8.png){width="6.0in" height="3.2618055555555556in"}
