"""
填充AI代码审查与修复表Word模板
作业4B：中频短线量化全流程（AI辅助版）
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from docx import Document
from docx.shared import Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH

# 模板路径
TEMPLATE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    '..', '作业四-B', 'AI代码审查与修复表 学生表.docx'
)
# 输出路径
OUTPUT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'outputs', 'homework4b'
)
OUTPUT_PATH = os.path.join(OUTPUT_DIR, '202331060205_丁致宇_AI代码审查与修复表.docx')


def fill_table_cell(cell, text, bold=False):
    """填充表格单元格"""
    cell.text = ''
    p = cell.paragraphs[0]
    run = p.add_run(text)
    run.font.size = Pt(10)
    if bold:
        run.bold = True


def main():
    # 读取模板
    doc = Document(TEMPLATE_PATH)

    # ==================== 表格1：基础信息 ====================
    table1 = doc.tables[0]

    # 学生姓名
    fill_table_cell(table1.cell(1, 1), '丁致宇')
    # 学号
    fill_table_cell(table1.cell(2, 1), '202331060205')
    # 作业名称
    fill_table_cell(table1.cell(3, 1), '作业4B：中频短线量化全流程（AI辅助版）')
    # 策略周期 - 选择日频短线
    fill_table_cell(table1.cell(4, 1), '☑ 日频短线 □ 周频中线 □ 混合周期')
    # 代码来源 - AI生成+人工修改
    fill_table_cell(table1.cell(5, 1), '□ 纯AI生成 ☑ AI生成+人工修改 □ 人工自主编写')
    # AI生成内容占比
    fill_table_cell(table1.cell(6, 1), '约70%')
    # 策略核心因子类型
    fill_table_cell(table1.cell(7, 1), '□ 纯快变量 ☑ 快变量为主+慢变量过滤 □ 快慢变量混合')

    # ==================== 表格2：模块1 时序逻辑审查 ====================
    table2 = doc.tables[1]

    # 1.1 无未来函数
    fill_table_cell(table2.cell(1, 2), 'data.py:compute_factors, line 117')
    fill_table_cell(table2.cell(1, 3),
        'Reversal因子使用当日ret（-当日收益率），属于当日信号当日收益。'
        '但回测中信号在收盘产生，次日执行交易，无未来函数。')
    fill_table_cell(table2.cell(1, 4), '无')
    fill_table_cell(table2.cell(1, 5), '已修复')
    fill_table_cell(table2.cell(1, 6),
        'Reversal因子定义为"-当日个股收益率"是作业要求，'
        '实际回测中信号在收盘产生，次日执行交易，时间上无未来函数。')

    # 1.2 时间戳严格递增
    fill_table_cell(table2.cell(2, 2), 'data.py:load_parquet_data, line 44')
    fill_table_cell(table2.cell(2, 3),
        '所有数据加载后按trade_date排序，保证时间顺序')
    fill_table_cell(table2.cell(2, 4), '无')
    fill_table_cell(table2.cell(2, 5), '合规')
    fill_table_cell(table2.cell(2, 6),
        'df = df.sort_values([\'ts_code\', \'trade_date\'])')

    # 1.3 信号与交易分离
    fill_table_cell(table2.cell(3, 2), 'backtest.py:run_backtest, line 84-139')
    fill_table_cell(table2.cell(3, 3),
        '当日得分→收盘买入→次日卖出，信号与执行分离')
    fill_table_cell(table2.cell(3, 4), '无')
    fill_table_cell(table2.cell(3, 5), '合规')
    fill_table_cell(table2.cell(3, 6),
        '流程：打分→选股→T+1持有→T+2卖出')

    # 1.4 样本数据严格隔离
    fill_table_cell(table2.cell(4, 2), 'data.py:split_dataset, line 197-216')
    fill_table_cell(table2.cell(4, 3),
        '训练集(2020-2023)和回测集(2024-2025)完全分离')
    fill_table_cell(table2.cell(4, 4), '无')
    fill_table_cell(table2.cell(4, 5), '合规')
    fill_table_cell(table2.cell(4, 6),
        '严格按时间切分，无交叉')

    # 1.5 漂移测试
    fill_table_cell(table2.cell(5, 2), 'models.py:train_lgbm_with_cv, line 44-141')
    fill_table_cell(table2.cell(5, 3),
        '使用时间序列交叉验证，避免未来数据泄露')
    fill_table_cell(table2.cell(5, 4), '无')
    fill_table_cell(table2.cell(5, 5), '合规')
    fill_table_cell(table2.cell(5, 6),
        '5折时间序列CV，验证集在训练集之后')

    # ==================== 表格3：模块2 因子与数据处理审查 ====================
    table3 = doc.tables[2]

    # 2.1 快变量筛选合理
    fill_table_cell(table3.cell(1, 2), 'data.py:compute_factors, line 112-153')
    fill_table_cell(table3.cell(1, 3),
        '4个标准因子均为日频快变量，匹配1~2日持仓')
    fill_table_cell(table3.cell(1, 4), '无')
    fill_table_cell(table3.cell(1, 5), '合规')
    fill_table_cell(table3.cell(1, 6),
        'Reversal/Liquidity/MoneyFlow/Value均为日频')

    # 2.2 慢变量使用合规
    fill_table_cell(table3.cell(2, 2), 'data.py:compute_factors, line 132-134')
    fill_table_cell(table3.cell(2, 3),
        'Value(PE_TTM倒数)为慢变量，仅作为选股辅助')
    fill_table_cell(table3.cell(2, 4), '无')
    fill_table_cell(table3.cell(2, 5), '合规')
    fill_table_cell(table3.cell(2, 6),
        '慢变量不作为短线主信号')

    # 2.3 高频数据去噪完整
    fill_table_cell(table3.cell(3, 2), 'data.py:clean_data, line 156-193')
    fill_table_cell(table3.cell(3, 3),
        '使用3σ缩尾去极值，前值填充缺失值')
    fill_table_cell(table3.cell(3, 4), '无')
    fill_table_cell(table3.cell(3, 5), '合规')
    fill_table_cell(table3.cell(3, 6),
        'df[col].clip(lower, upper)')

    # 2.4 无因子共线性
    fill_table_cell(table3.cell(4, 2), 'factors.py:compute_vif, line 134-187')
    fill_table_cell(table3.cell(4, 3),
        '所有因子VIF < 2.3，远低于阈值5')
    fill_table_cell(table3.cell(4, 4), '无')
    fill_table_cell(table3.cell(4, 5), '合规')
    fill_table_cell(table3.cell(4, 6),
        'VIF最大为Momentum的2.23')

    # 2.5 因子经济逻辑清晰
    fill_table_cell(table3.cell(5, 2), 'data.py:compute_factors, line 112-153')
    fill_table_cell(table3.cell(5, 3),
        '所有因子均有明确金融逻辑支撑')
    fill_table_cell(table3.cell(5, 4), '无')
    fill_table_cell(table3.cell(5, 5), '合规')
    fill_table_cell(table3.cell(5, 6),
        '反转、流动性、资金流、价值均有理论依据')

    # ==================== 表格4：模块3 交易与风控审查 ====================
    table4 = doc.tables[3]

    # 3.1 交易成本完整
    fill_table_cell(table4.cell(1, 2), 'config.py:COMMISSION_RATE, line 73')
    fill_table_cell(table4.cell(1, 3),
        '单边0.1%，双边0.2%，含佣金+滑点')
    fill_table_cell(table4.cell(1, 4), 'P1')
    fill_table_cell(table4.cell(1, 5), '合规')
    fill_table_cell(table4.cell(1, 6),
        '考虑A股无资本利得税，0.1%单边合理')

    # 3.2 仓位合规
    fill_table_cell(table4.cell(2, 2), 'config.py:TOP_N_STOCKS, line 75')
    fill_table_cell(table4.cell(2, 3),
        '5只等权，单票20%，总仓位100%')
    fill_table_cell(table4.cell(2, 4), 'P1')
    fill_table_cell(table4.cell(2, 5), '部分修复')
    fill_table_cell(table4.cell(2, 6),
        '总仓位100%满仓，未设置80%上限。作业要求选Top5等权，符合要求。')

    # 3.3 止损机制完善
    fill_table_cell(table4.cell(3, 2), 'backtest.py')
    fill_table_cell(table4.cell(3, 3),
        '无显式止损机制，依赖因子模型自动调仓')
    fill_table_cell(table4.cell(3, 4), 'P2')
    fill_table_cell(table4.cell(3, 5), '待优化')
    fill_table_cell(table4.cell(3, 6),
        '当前依赖LGBM模型自动选股实现风控')

    # 3.4 流动性过滤
    fill_table_cell(table4.cell(4, 2), 'data.py:load_parquet_data, line 31-32')
    fill_table_cell(table4.cell(4, 3),
        '沪深300成分股天然具有高流动性')
    fill_table_cell(table4.cell(4, 4), '无')
    fill_table_cell(table4.cell(4, 5), '合规')
    fill_table_cell(table4.cell(4, 6),
        '成分股为A股流动性最优标的')

    # 3.5 调仓频率合规
    fill_table_cell(table4.cell(5, 2), 'config.py:HOLDING_DAYS, line 76')
    fill_table_cell(table4.cell(5, 3),
        '每日调仓，平均持仓1~2日')
    fill_table_cell(table4.cell(5, 4), '无')
    fill_table_cell(table4.cell(5, 5), '合规')
    fill_table_cell(table4.cell(5, 6),
        '符合中频1~2日持仓要求')

    # ==================== 表格5：模块4 AI代码特有缺陷审查 ====================
    table5 = doc.tables[4]

    # 4.1 无AI幻觉
    fill_table_cell(table5.cell(1, 2), '全局')
    fill_table_cell(table5.cell(1, 3),
        '所有import的库均真实存在（pandas, numpy, lightgbm, scipy等）')
    fill_table_cell(table5.cell(1, 4), '无')
    fill_table_cell(table5.cell(1, 5), '合规')
    fill_table_cell(table5.cell(1, 6),
        '已验证所有依赖版本兼容')

    # 4.2 无逻辑矛盾
    fill_table_cell(table5.cell(2, 2), 'backtest.py:run_backtest, line 40-157')
    fill_table_cell(table5.cell(2, 3),
        '条件判断清晰，交易方向明确')
    fill_table_cell(table5.cell(2, 4), '无')
    fill_table_cell(table5.cell(2, 5), '合规')
    fill_table_cell(table5.cell(2, 6),
        '买入/卖出逻辑无冲突')

    # 4.3 无过度拟合
    fill_table_cell(table5.cell(3, 2), 'models.py:train_lgbm_with_cv, line 44-141')
    fill_table_cell(table5.cell(3, 3),
        '使用时间序列CV，早停机制防止过拟合')
    fill_table_cell(table5.cell(3, 4), '无')
    fill_table_cell(table5.cell(3, 5), '合规')
    fill_table_cell(table5.cell(3, 6),
        '早停20轮，最佳迭代次数9-41')

    # 4.4 无硬编码参数
    fill_table_cell(table5.cell(4, 2), 'config.py')
    fill_table_cell(table5.cell(4, 3),
        '所有关键参数均可配置')
    fill_table_cell(table5.cell(4, 4), '无')
    fill_table_cell(table5.cell(4, 5), '合规')
    fill_table_cell(table5.cell(4, 6),
        '集中在config.py管理')

    # 4.5 无冗余代码
    fill_table_cell(table5.cell(5, 2), '全局')
    fill_table_cell(table5.cell(5, 3),
        '代码结构清晰，无重复逻辑')
    fill_table_cell(table5.cell(5, 4), '无')
    fill_table_cell(table5.cell(5, 5), '合规')
    fill_table_cell(table5.cell(5, 6),
        '模块化设计，职责分离')

    # ==================== 表格6：模块5 代码可读性审查 ====================
    table6 = doc.tables[5]

    # 5.1 注释完整
    fill_table_cell(table6.cell(1, 2), '全局')
    fill_table_cell(table6.cell(1, 3),
        '核心逻辑、因子计算、交易规则均有清晰中文注释')
    fill_table_cell(table6.cell(1, 4), '无')
    fill_table_cell(table6.cell(1, 5), '合规')
    fill_table_cell(table6.cell(1, 6),
        '每个模块和函数均有docstring')

    # 5.2 命名规范
    fill_table_cell(table6.cell(2, 2), '全局')
    fill_table_cell(table6.cell(2, 3),
        '变量、函数命名符合Python规范，见名知意')
    fill_table_cell(table6.cell(2, 4), '无')
    fill_table_cell(table6.cell(2, 5), '合规')
    fill_table_cell(table6.cell(2, 6),
        '使用snake_case命名')

    # 5.3 结构清晰
    fill_table_cell(table6.cell(3, 2), '全局')
    fill_table_cell(table6.cell(3, 3),
        '按数据→因子→质检→模型→回测→绘图模块化拆分')
    fill_table_cell(table6.cell(3, 4), '无')
    fill_table_cell(table6.cell(3, 5), '合规')
    fill_table_cell(table6.cell(3, 6),
        '7个独立模块+main.py入口')

    # 5.4 错误处理完善
    fill_table_cell(table6.cell(4, 2), 'data.py, factors.py')
    fill_table_cell(table6.cell(4, 3),
        'try-except捕获异常，NaN检查')
    fill_table_cell(table6.cell(4, 4), 'P2')
    fill_table_cell(table6.cell(4, 5), '部分修复')
    fill_table_cell(table6.cell(4, 6),
        '部分函数缺少完整的错误处理')

    # 5.5 结果可复现
    fill_table_cell(table6.cell(5, 2), 'config.py:LGBM_PARAMS, line 65')
    fill_table_cell(table6.cell(5, 3),
        '固定随机种子seed=42')
    fill_table_cell(table6.cell(5, 4), '无')
    fill_table_cell(table6.cell(5, 5), '合规')
    fill_table_cell(table6.cell(5, 6),
        'LGBM参数中设置seed=42')

    # ==================== 表格7：整体评估 ====================
    table7 = doc.tables[6]

    # 核心缺陷TOP3
    fill_table_cell(table7.cell(1, 1),
        '1. 无显式止损机制：回测依赖模型自动选股，未设置单笔止损和账户总回撤控制\n'
        '2. 总仓位100%满仓：未设置80%仓位上限，极端行情下风险敞口过大\n'
        '3. 错误处理不完善：部分函数缺少完整的try-except和日志记录')

    # 人工修复核心亮点
    fill_table_cell(table7.cell(2, 1),
        '1. MoneyFlow因子修复：添加5日滚动净流入计算，符合作业要求的定义\n'
        '2. 回测逻辑修复：修正了持仓收益计算的时间对齐bug，确保无未来函数\n'
        '3. 扩展因子添加：增加Momentum/Volatility/Turnover/VolumeChange四个因子增强模型能力')

    # 合规性结论
    fill_table_cell(table7.cell(3, 1),
        '□ 完全合规（所有P0/P1问题已修复）\n'
        '☑ 基本合规（仅剩余低优先级优化问题）')

    # 总结反思
    fill_table_cell(table7.cell(4, 1),
        '1. AI擅长快速实现标准化流程（数据加载、因子计算、模型训练），'
        '但因子经济逻辑仍需人工判断\n'
        '2. 回测代码的时间对齐和未来函数检查需要人工仔细审核，'
        'AI容易产生微妙的时序错误\n'
        '3. 模型可解释性不足，LGBM特征重要性仅反映统计关系，非因果关系\n'
        '4. AI生成的代码可能包含逻辑矛盾（如同时买入和卖出），需要逐行审查')

    # 保存文档
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    doc.save(OUTPUT_PATH)
    print(f'AI代码审查与修复表已保存至: {OUTPUT_PATH}')


if __name__ == '__main__':
    main()
