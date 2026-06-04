"""
作业4B PPT生成脚本
生成《中频短线量化全流程（AI辅助版）》演示文稿
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.enum.text import PP_ALIGN
from pptx.dml.color import RGBColor

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'outputs', 'homework4b')
PPT_PATH = os.path.join(OUTPUT_DIR, '202331060205_丁致宇_作业4B.pptx')


def add_title_slide(prs, title, subtitle):
    slide = prs.slides.add_slide(prs.slide_layouts[0])
    slide.shapes.title.text = title
    slide.placeholders[1].text = subtitle
    for para in slide.placeholders[1].text_frame.paragraphs:
        para.font.size = Pt(18)
    return slide


def add_content_slide(prs, title, content_lines):
    slide = prs.slides.add_slide(prs.slide_layouts[1])
    slide.shapes.title.text = title
    tf = slide.placeholders[1].text_frame
    tf.clear()
    for i, line in enumerate(content_lines):
        if i == 0:
            p = tf.paragraphs[0]
        else:
            p = tf.add_paragraph()
        p.text = line
        p.font.size = Pt(14)
        p.space_after = Pt(6)
        if line.startswith('##') or line.startswith('**'):
            p.font.bold = True
            p.font.size = Pt(16)
    return slide


def add_image_slide(prs, title, img_path, width=Inches(9)):
    slide = prs.slides.add_slide(prs.slide_layouts[5])
    slide.shapes.title.text = title
    if os.path.exists(img_path):
        slide.shapes.add_picture(img_path, Inches(0.5), Inches(1.5), width=width)
    return slide


def main():
    prs = Presentation()
    prs.slide_width = Inches(13.33)
    prs.slide_height = Inches(7.5)

    # ========== 第1页：封面 ==========
    add_title_slide(prs,
        '作业4B：中频短线量化全流程（AI辅助版）',
        '因子挖掘→IC/共线质检→机器学习赋权→1~2日持仓选股回测\n\n'
        '学生：丁致宇  学号：202331060205\n'
        '课程：金融与经济数据挖掘\n'
        '日期：2026年6月')

    # ========== 第2页：作业概述 ==========
    add_content_slide(prs, '一、作业概述', [
        '**核心目标**',
        '基于沪深300成分股构建中频短线量化模型',
        '掌握日频因子筛选、AI辅助编程、中频交易规则与回测',
        '',
        '**研究设计**',
        '• 研究标的：沪深300指数成分股（300只A股）',
        '• 训练集：2020-01-01 ~ 2023-12-31（因子检验、模型训练）',
        '• 回测集：2024-01-01 ~ 2025-12-31（样本外验证）',
        '',
        '**核心流程**',
        '本地数据加载→数据预处理→中频因子计算→IC/IR+VIF共线质检',
        '→单因子检验→LGBM机器学习赋权→日频截面选股→1~2日持仓回测',
    ])

    # ========== 第3页：数据概述 ==========
    add_content_slide(prs, '二、数据概述', [
        '**数据来源：** 教师预下载的本地Parquet格式数据（Tushare Pro）',
        '',
        '**数据结构：**',
        '• daily/ — 个股日线行情（300只股票，含OHLCV）',
        '• daily_basic/ — 个股日线基本面（PE_TTM、流通市值、换手率等）',
        '• index_daily/ — 沪深300指数日线',
        '• finance/ — 个股财务数据',
        '• 5min/ — 个股5分钟行情',
        '• moneyflow/ — 沪深港通资金流向',
        '',
        '**数据规模：** 299只股票，约48万条日频记录（2019-2025）',
    ])

    # ========== 第4页：因子定义 ==========
    add_content_slide(prs, '三、因子定义', [
        '**四大标准因子（作业要求）：**',
        '',
        '• Reversal（反转）：-前1日个股收益率',
        '  → 超短反转预测，捕捉短期价格回归',
        '',
        '• Liquidity（流动性）：|日收益率| / 日成交额',
        '  → Amihud非流动性指标，衡量交易摩擦',
        '',
        '• MoneyFlow（资金流）：成交额×价格方向 / 流通市值',
        '  → 资金流入流出代理（无个股级buy/sell数据）',
        '',
        '• Value（价值）：1 / PE_TTM',
        '  → 低估值正向化，增强收益稳定性',
        '',
        '**四个扩展因子（增强模型能力）：**',
        '• Momentum（动量）：5日累计收益',
        '• Volatility（波动率）：5日收益标准差',
        '• Turnover（换手率）：日换手率',
        '• VolumeChange（量比）：当日成交量/5日均量',
    ])

    # ========== 第5页：因子质检 - IC/IR ==========
    add_image_slide(prs, '四、因子质检：IC/IR分析',
        os.path.join(OUTPUT_DIR, 'ic_ir_summary.png'), Inches(10))

    # ========== 第6页：IC序列图 ==========
    add_image_slide(prs, '四、因子质检：IC时间序列',
        os.path.join(OUTPUT_DIR, 'ic_series.png'), Inches(10))

    # ========== 第7页：因子质检结果表 ==========
    add_content_slide(prs, '四、因子质检：结果汇总', [
        '**IC/IR检验结果：**',
        '',
        '因子           IC均值     IR值     判定',
        '─────────────────────────────────────',
        'Reversal      0.0297    0.1613   有效 ✓',
        'Liquidity     0.0148    0.1242   边缘',
        'MoneyFlow    -0.0317   -0.1829   有效 ✓（负向）',
        'Value         0.0160    0.0645   边缘',
        'Momentum     -0.0243   -0.1227   有效 ✓（负向）',
        'Volatility   -0.0243   -0.1204   有效 ✓（负向）',
        'Turnover     -0.0293   -0.1253   有效 ✓（负向）',
        'VolumeChange -0.0093   -0.0720   无效',
        '',
        '**VIF共线性检验：** 所有因子VIF < 5，无高共线问题',
        '',
        '**单因子OLS：** 所有因子p > 0.05（线性维度失效，A股常态）',
        '→ 全部顺延进入LGBM非线性二次筛选',
    ])

    # ========== 第8页：VIF结果 ==========
    add_content_slide(prs, '四、因子质检：VIF共线性检验', [
        '**VIF检验结果（VIF > 5判定高共线）：**',
        '',
        '因子           VIF值     判定',
        '─────────────────────────────',
        'Reversal      2.1914    正常',
        'Liquidity     1.0401    正常',
        'MoneyFlow     1.9052    正常',
        'Value         1.1692    正常',
        'Momentum      1.3241    正常',
        'Volatility    1.6035    正常',
        'Turnover      1.6052    正常',
        'VolumeChange  1.1715    正常',
        '',
        '**结论：** 所有因子VIF均小于5，因子间无严重共线性，',
        '可全部纳入LGBM模型训练。',
    ])

    # ========== 第9页：LGBM模型 ==========
    add_content_slide(prs, '五、LGBM机器学习赋权', [
        '**模型配置：**',
        '• 模型：LightGBM回归模型（预测次日超额收益）',
        '• 训练集：2020-2023年（约27万条样本）',
        '• 交叉验证：5折时间序列交叉验证',
        '• 早停轮数：20轮',
        '',
        '**交叉验证结果：**',
        '• 最佳折：Fold 5，验证MSE = 0.000473',
        '• 训练MSE：0.000738',
        '• 模型自动学习因子非线性权重',
        '',
        '**模型特点：**',
        '• 替代传统手工加权综合得分',
        '• 自动捕捉因子间的非线性交互关系',
        '• 特征重要性由模型训练过程中的增益决定',
    ])

    # ========== 第10页：特征重要性 ==========
    add_image_slide(prs, '五、LGBM模型：因子特征重要性',
        os.path.join(OUTPUT_DIR, 'factor_weights.png'), Inches(10))

    # ========== 第11页：选股规则 ==========
    add_content_slide(prs, '六、日频截面选股与交易规则', [
        '**综合得分定义：**',
        '• 综合得分 = LGBM模型预测的个股次日超额收益',
        '• 得分越高，1~2日持仓收益预期越强',
        '',
        '**选股规则：**',
        '• 每日截面打分，按综合得分从高到低排序',
        '• 选Top5股票，等权持仓（每只20%）',
        '',
        '**交易规则（中频1~2日核心）：**',
        '• 调仓频率：每日调仓（平均持仓1~2日）',
        '• 仓位管理：5只等权，单票≤20%',
        '• 交易流程：卖出旧持仓→买入新持仓',
        '• 交易成本：单边0.1%（含佣金+印花税+滑点）',
        '',
        '**风控措施：**',
        '• 3σ缩尾去极值处理因子异常值',
        '• 每日收益率剔除停牌日',
        '• 流通市值过滤低流动性标的',
    ])

    # ========== 第12页：回测结果 ==========
    add_image_slide(prs, '七、回测结果：策略净值 vs 沪深300',
        os.path.join(OUTPUT_DIR, 'nav_vs_market.png'), Inches(10))

    # ========== 第13页：回撤曲线 ==========
    add_image_slide(prs, '七、回测结果：回撤曲线',
        os.path.join(OUTPUT_DIR, 'drawdown.png'), Inches(10))

    # ========== 第14页：核心绩效指标 ==========
    add_image_slide(prs, '七、回测核心绩效指标',
        os.path.join(OUTPUT_DIR, 'metrics_table.png'), Inches(9))

    # ========== 第15页：回测指标解读 ==========
    add_content_slide(prs, '七、回测指标解读', [
        '**核心指标分析：**',
        '',
        '• 累计收益率：236.21%（回测期2年）',
        '• 年化收益率：88.26%（远超沪深300的17.2%）',
        '• 夏普比率：1.5837（风险调整后收益优秀）',
        '• 最大回撤：-30.60%（风险可控）',
        '• 日胜率：47.00%（盈亏比合理）',
        '• 超额收益（累计）：200.65%（大幅跑赢基准）',
        '• 信息比率：1.5787（超额收益稳定性好）',
        '• Calmar比率：2.8847（收益/回撤比优秀）',
        '',
        '**结论：** 策略在样本外（2024-2025）表现稳健，',
        '年化超额收益71.05%，信息比率>1.5，',
        '验证了LGBM非线性因子赋权的有效性。',
    ])

    # ========== 第16页：总结与反思 ==========
    add_content_slide(prs, '八、总结与反思', [
        '**核心发现：**',
        '',
        '1. IC/IR分析揭示：Reversal和MoneyFlow是A股短线最有效的因子',
        '   但单因子OLS均为p>0.05（线性维度失效），验证了A股因子的非线性特征',
        '',
        '2. LGBM模型自动学习到：Liquidity、Volatility、VolumeChange',
        '   在非线性交互中贡献最大，而非传统线性有效的因子',
        '',
        '3. 日频调仓及时响应模型信号变化',
        '   虽然交易成本较高，但信号更新带来的收益超过成本',
        '',
        '**AI辅助编程的局限性反思：**',
        '• AI擅长快速实现标准化流程，但因子经济逻辑仍需人工判断',
        '• 回测结果需人工审核，避免过拟合和未来函数',
        '• 模型可解释性不足，特征重要性仅反映统计关系，非因果关系',
    ])

    # ========== 第17页：提交材料清单 ==========
    add_content_slide(prs, '九、提交材料清单', [
        '**提交内容：**',
        '',
        '1. 代码文件（homework4b/目录）：',
        '   config.py - 配置参数',
        '   data.py - 数据加载与因子计算',
        '   factors.py - 因子质检（IC/IR/VIF/OLS）',
        '   models.py - LGBM模型训练',
        '   backtest.py - 回测引擎',
        '   plots.py - 可视化',
        '   main.py - 主程序入口',
        '',
        '2. 输出结果（outputs/homework4b/）：',
        '   summary.json - 回测指标',
        '   6张图表（净值对比、回撤、IC、IR、特征重要性、指标表）',
        '',
        '3. 文档：',
        '   AI代码审核记录',
        '   AI交互记录',
        '   本PPT演示文稿',
    ])

    # 保存
    prs.save(PPT_PATH)
    print(f"PPT已保存至: {PPT_PATH}")
    print(f"共{len(prs.slides)}页")


if __name__ == '__main__':
    main()
