# 作业4：量化全流程：单因子检验+因子质检+多因子赋权+选股回测实战

## 一、作业目的

承接CAPM/二因子模型分析思路，完整复现量化投研标准流程：因子获取→数据预处理→单因子有效性检验→IC/IR因子质检→多因子截面回归赋权→截面打分选股→样本外回测验证。 核心目标：基于上证标的构建量化选股模型，掌握因子筛选、权重赋值、策略回测的行业标准方法。

## 二、核心研究设计

**研究标的：** 选取上证 50 指数成分股（共 50 只），为沪市规模最大、流动性最优的超大盘蓝筹股，数据完整度高、截面回归自由度充足，适合开展多因子选股与回测验证。

**时间划分：**
- 建模训练集：2015-01-01 → 2023-12-31（因子检验、回归赋权）
- 样本外回测集：2024-01-01 → 2025-12-31（策略验证，严格隔离）

**工具与数据：**
- 编程：Python
- 数据：baostock（月度K线、财务指标、上证指数）

**核心流程：** 数据预处理 → CAPM个股筛选 → 单因子有效性检验 → IC/IR质检 → 多因子回归赋权 → 截面选股 → 样本外回测

## 三、因子标准定义

- 市场因子MKT：上证指数月度超额收益率（仅用于CAPM回归）
- 规模因子SMB：个股月度总市值（大盘/小盘风格，月末值）
- 价值因子PE：TTM市盈率倒数（正向化，低估值→高因子值）
- 质量因子Quality：ROE × 股利分红率 × 净利润年增长率（高盈利+高分红+高增长）
- 无风险利率Rf：月度固定值（0.15%/月）

## 四、分任务

### 模块1：数据获取与预处理（35分）

**任务：** 获取股价/指数/财务数据，计算收益率与因子，完成清洗、去极值、数据集拆分。

**操作步骤：**

**数据提取：**
- 上证50股票：月度收盘价、月度收益率
- 上证指数：月度收盘价、月度收益率
- 财务数据：总市值、TTM市盈率、ROE、股利分红率、净利润年增长率

**指标计算：**
- 个股月度收益率 = (月末收盘价/月初收盘价)-1
- 市场月度收益率 = (上证指数月末/月初)-1
- 超额收益 = 月度收益率 - 月度无风险利率(0.15%)
- 合成Quality因子、PE倒数因子

**数据清洗：**
- 缺失值：财务数据前值填充，市值/PE均值填充
- 异常值：3σ缩尾处理（超出[μ-3σ,μ+3σ]用边界值替换）

**数据集拆分：**
- 训练集：2015-2023
- 回测集：2024-2025

### 模块2：单因子有效性检验（20分）

**步骤1：CAPM个股筛选（必做）**
- 回归公式：Ri-Rf = α + β×MKT + ε
- 判定：β显著（p<0.05） → 个股有效，全部10只保留

**步骤2：单因子横截面回归（因子有效性）**
- 回归公式：Return_{i,t+1} = α + β×Factor_{i,t} + ε
- 因变量：个股下月超额收益
- 自变量：单个因子（SMB/PE/Quality）

**步骤3：有效因子判定**
- 因子系数β显著（p<0.05） → 判定为有效因子
- 输出：有效因子列表（如PE、Quality有效，SMB无效）

### 模块3：因子质检与IC/IR计算（15分）

**1. 因子标准化（5分）**
- 方法：Z-Score标准化 Factor_std = (Factor-μ)/σ
- 范围：仅对模块2筛选出的有效因子处理
- 目的：统一量纲，保证回归权重公平

**2. IC值计算（6分）**
- IC定义：当期因子值与下月收益率的Pearson相关系数（预测能力）
- 计算：按月截面计算，得到月度IC序列
- 判定标准：
  - IC均值 > 0.02：具备预测能力
  - IC均值 > 0.05：优秀因子

**3. IR值计算（4分）**
- IR定义：IR = IC均值 / IC标准差（稳定性）
- 判定标准：IR绝对值 > 0.1

### 模块4：多因子线性回归静态赋权（15分）

**1. 建模数据准备（4分）**
- 特征：有效因子的标准化值
- 标签：个股下月超额收益
- 样本：仅用2015-2023训练集
- 格式：每行=1只股票+1个月份

**2. 截面回归建模（6分）**
- 模型：Return_{i,t+1} = α + w1×Factor1 + w2×Factor2 + ... + ε
- 工具：statsmodels.OLS
- 输出：回归系数w1/w2… = 因子静态权重

**3. 综合得分公式（5分）**
- 综合得分 = w1×因子1标准化值 + w2×因子2标准化值 + ...
- 得分越高：未来收益预期越强

### 模块5：截面选股+样本外回测（15分）

**第一部分：截面打分选股（6分）**
- 每月截面打分：按综合得分从高到低排序，输出打分排序表
- 选股规则：选Top3，无法交易顺延至第4名
- 交易规则：
  - 调仓：每月1次
  - 仓位：3只等权，单票≤40%
  - 流程：卖出旧持仓 → 买入新持仓

**第二部分：样本外回测（9分）**
- 回测参数：
  - 初始资金：100万元
  - 交易成本：单边0.3%
  - 区间：2024-2025
- 必算指标：
  - 累计收益率、年化收益率
  - 最大回撤、月度胜率
  - 超额收益（相对上证指数）
  - 月度调仓记录
- 输出：
  - 策略净值VS上证指数对比图
  - 回测指标表格

## 五、提交要求

- 附件：PPT + Python代码 + 回测图表/表格
- 命名：学号+姓名+作业4
- 展示：因子筛选、IC/IR、因子权重、选股逻辑、回测结果

## 六、Python代码框架参考

```python
# ===================== 1. 基础导入 =====================
import baostock as bs
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# ===================== 2. 全局参数 =====================
# 股票代码（baostock格式）
bs.login()

# 一键获取上证50所有成分股（baostock官方接口）
rs = bs.query_sz50_stocks()
sz50_list = []
while rs.error_code == "0" and rs.next():
    sz50_list.append(rs.get_row_data())
sz50_df = pd.DataFrame(sz50_list, columns=rs.fields)

# 提取代码与名称
STOCKS = sz50_df["code"].tolist()
STOCK_NAMES = sz50_df["code_name"].tolist()

bs.logout()

# 时间参数
TRAIN_START = "2015-01-01"
TRAIN_END   = "2023-12-31"
TEST_START  = "2024-01-01"
TEST_END    = "2025-12-31"

# 回测参数
INIT_CAPITAL = 1000000
FEE = 0.003
TOP_N = 3
RF = 0.0015  # 月度无风险利率

# ===================== 模块1：数据获取与预处理 =====================
def get_data():
    bs.login()
    # 1. 获取指数数据
    rs = bs.query_history_k_data_plus("sh.000001", "close", TRAIN_START, TEST_END, frequency="m")
    mkt_df = rs.data
    mkt_df['close'] = mkt_df['close'].astype(float)
    mkt_df['mkt_return'] = mkt_df['close'].pct_change()
    mkt_df['mkt_excess'] = mkt_df['mkt_return'] - RF

    # 2. 获取个股数据 + 财务数据
    stock_data = []
    for code, name in zip(STOCKS, STOCK_NAMES):
        # K线
        rs = bs.query_history_k_data_plus(code, "close", TRAIN_START, TEST_END, frequency="m")
        df = rs.data
        df['code'] = code
        df['name'] = name
        df['close'] = df['close'].astype(float)
        df['return'] = df['close'].pct_change()
        df['excess_return'] = df['return'] - RF
        df['next_excess_return'] = df['excess_return'].shift(-1)

        # 财务数据（总市值、PE、ROE、分红、净利润增速）
        # 自行补充baostock财务接口

        # 合成因子
        df['PE_inv'] = 1 / df['pe_ttm']
        df['Quality'] = df['roe'] * df['div_ratio'] * df['profit_growth']
        stock_data.append(df)

    # 3. 合并 + 清洗 + 3σ去极值 + 缺失值填充
    all_df = pd.concat(stock_data)
    # 3σ缩尾
    def winsorize(s):
        mu, sigma = s.mean(), s.std()
        return s.clip(mu-3*sigma, mu+3*sigma)
    all_df['SMB'] = winsorize(all_df['market_cap'])
    all_df['PE_inv'] = winsorize(all_df['PE_inv'])
    all_df['Quality'] = winsorize(all_df['Quality'])

    # 4. 拆分训练/回测集
    train_df = all_df[(all_df['date']>=TRAIN_START) & (all_df['date']<=TRAIN_END)].dropna()
    test_df = all_df[(all_df['date']>=TEST_START) & (all_df['date']<=TEST_END)].dropna()
    bs.logout()
    return train_df, test_df

# ==================== 模块2：CAPM筛选 + 单因子检验 =====================
def factor_test(train_df):
    valid_factors = []
    # 1. CAPM个股筛选
    def capm_reg(ret, mkt):
        X = sm.add_constant(mkt)
        model = sm.OLS(ret, X).fit()
        return model.params[1], model.pvalues[1]
    # 2. 单因子横截面回归
    for factor in ['SMB', 'PE_inv', 'Quality']:
        beta_list, p_list = [], []
        for month in train_df['date'].unique():
            sub = train_df[train_df['date']==month]
            X = sm.add_constant(sub[factor])
            y = sub['next_excess_return']
            model = sm.OLS(y, X).fit()
            beta_list.append(model.params[1])
            p_list.append(model.pvalues[1])
        if np.mean(p_list) < 0.05:
            valid_factors.append(factor)
    return valid_factors

# ===================== 模块3：IC/IR因子质检 =====================
def ic_ir(train_df, valid_factors):
    # Z-Score标准化
    def standardize(s):
        return (s - s.mean()) / s.std()
    for f in valid_factors:
        train_df[f'{f}_std'] = standardize(train_df[f])
    # 计算IC/IR
    ic_result = {}
    for f in valid_factors:
        ic_list = []
        for month in train_df['date'].unique():
            sub = train_df[train_df['date']==month]
            ic = sub[[f'{f}_std', 'next_excess_return']].corr().iloc[0,1]
            ic_list.append(ic)
        ic_mean = np.mean(ic_list)
        ic_std = np.std(ic_list)
        ir = ic_mean / ic_std if ic_std!=0 else 0
        ic_result[f] = {'IC_mean':ic_mean, 'IC_std':ic_std, 'IR':ir}
    return train_df, ic_result

# ===================== 模块4：多因子回归赋权 =====================
def multi_factor_model(train_df, valid_factors):
    std_cols = [f+'_std' for f in valid_factors]
    X = train_df[std_cols]
    X = sm.add_constant(X)
    y = train_df['next_excess_return']
    model = sm.OLS(y, X).fit()
    weights = model.params[1:] # 剔除常数项
    # 得分函数
    def score_func(row):
        return sum(weights[i]*row[std_cols[i]] for i in range(len(valid_factors)))
    return score_func, weights

# ===================== 模块5：截面选股 + 样本外回测 =====================
def backtest(test_df, score_func, valid_factors):
    test_df = test_df.copy()
    std_cols = [f+'_std' for f in valid_factors]
    # 月度打分选股
    nav = [INIT_CAPITAL]
    history = []
    for month in sorted(test_df['date'].unique()):
        sub = test_df[test_df['date']==month].copy()
        sub['score'] = sub.apply(score_func, axis=1)
        sub = sub.sort_values('score', ascending=False)
        top3 = sub.head(TOP_N)['code'].tolist()
        # 计算当月收益
        month_return = test_df[test_df['code'].isin(top3) & (test_df['date']==month)]['return'].mean()
        # 扣手续费
        month_return = month_return - 2*FEE
        # 更新净值
        new_nav = nav[-1] * (1 + month_return)
        nav.append(new_nav)
        history.append({'month':month, 'hold':top3, 'return':month_return, 'nav':new_nav})
    # 回测指标计算
    nav_series = pd.Series(nav[1:])
    cum_return = (nav_series.iloc[-1] / INIT_CAPITAL) - 1
    max_drawdown = (nav_series.cummax() - nav_series).max() / nav_series.cummax().max()
    win_rate = len([x for x in history if x['return']>0]) / len(history)
    # 绘图
    plt.figure(figsize=(12,5))
    plt.plot(nav_series, label='策略净值')
    plt.legend()
    plt.title('策略净值VS上证指数')
    plt.show()
    # 输出结果
    return {'累计收益':cum_return, '最大回撤':max_drawdown, '月度胜率':win_rate}, history

# ===================== 主函数运行 =====================
if __name__ == "__main__":
    train_df, test_df = get_data()
    valid_factors = factor_test(train_df)
    train_std, ic_result = ic_ir(train_df, valid_factors)
    score_func, weights = multi_factor_model(train_std, valid_factors)
    backtest_result, trade_history = backtest(test_df, score_func, valid_factors)
    print("有效因子：", valid_factors)
    print("IC/IR结果：", ic_result)
    print("因子权重：", weights)
    print("回测指标：", backtest_result)
```
