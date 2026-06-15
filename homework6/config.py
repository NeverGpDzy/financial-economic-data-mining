"""Configuration for Homework 6 FCFF value investing analysis."""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "homework6"
ORIGINAL_DIR = DATA_DIR / "original"
RAW_ROOT = DATA_DIR / "raw" / "作业6 配套数据" / "data"
OUTPUT_DIR = ROOT / "outputs" / "homework6"
REPORT_DIR = ROOT / "report"

STUDENT_NAME = "丁致宇"
STUDENT_ID = "202331060205"

# The assignment text says 2006-2025. The supplied data actually covers
# 2014-03-31 through 2024-12-31, so the reproducible口径 below follows the
# teacher-provided files and avoids fabricating unavailable years.
DATA_START_YEAR = 2014
DATA_END_YEAR = 2024
TRAIN_START_YEAR = 2014
TRAIN_END_YEAR = 2017
MODEL_TRAIN_END_YEAR = 2016  # labels are t+1, so this avoids using 2018 labels.
TEST_START_YEAR = 2018
TEST_END_YEAR = 2024
FCFF_3Y_TEST_END_YEAR = 2021

INITIAL_CAPITAL = 1_000_000.0
COMMISSION_RATE = 0.001
TRADING_DAYS = 252

IC_THRESHOLD = 0.02
IR_THRESHOLD = 0.10
VIF_THRESHOLD = 10.0
OLS_P_THRESHOLD = 0.05

GROUP_LABELS = ["A", "B", "C"]

FACTOR_META = {
    "F1_profit_cagr": {
        "label": "F1 年净利润复合增速",
        "direction": 1,
        "theme": "好商业模式",
    },
    "F2_gross_margin": {
        "label": "F2 销售毛利率",
        "direction": 1,
        "theme": "好商业模式",
    },
    "F3_net_margin": {
        "label": "F3 销售净利率",
        "direction": 1,
        "theme": "好商业模式",
    },
    "F4_light_asset": {
        "label": "F4 固资/总资产",
        "direction": -1,
        "theme": "好商业模式",
    },
    "F5_low_debt": {
        "label": "F5 资产负债率",
        "direction": -1,
        "theme": "好商业模式",
    },
    "F6_roe": {
        "label": "F6 年平均ROE",
        "direction": 1,
        "theme": "经济护城河",
    },
    "F7_low_expense": {
        "label": "F7 三费费率/营收",
        "direction": -1,
        "theme": "经济护城河",
    },
    "F8_ocf_profit": {
        "label": "F8 经营现金流净额/净利润",
        "direction": 1,
        "theme": "长期现金流",
    },
    "F9_dividend_payout": {
        "label": "F9 分红率",
        "direction": 1,
        "theme": "长期现金流",
    },
    "F10_dividend_yield": {
        "label": "F10 股息率",
        "direction": 1,
        "theme": "长期现金流",
    },
    "F11_fcff_yield": {
        "label": "F11 FCFF收益率",
        "direction": 1,
        "theme": "长期现金流",
    },
}

FACTOR_COLUMNS = list(FACTOR_META.keys())
FACTOR_Z_COLUMNS = [f"{col}_z" for col in FACTOR_COLUMNS]

TRADITIONAL_RULES = {
    "rule_profit_growth_positive": "F1_profit_cagr > 0",
    "rule_net_margin_positive": "F3_net_margin > 0",
    "rule_light_asset": "F4_light_asset < 0.35",
    "rule_debt_control": "F5_low_debt < 85",
    "rule_roe_quality": "F6_roe > 8",
    "rule_expense_control": "F7_low_expense < 35",
    "rule_cash_quality": "F8_ocf_profit > 50",
    "rule_dividend_positive": "F9_dividend_payout > 0.05",
    "rule_dividend_yield": "F10_dividend_yield > 0.5",
    "rule_fcff_positive": "fcff > 0",
}

LGBM_PARAMS = {
    "objective": "regression",
    "metric": "l2",
    "boosting_type": "gbdt",
    "max_depth": 3,
    "num_leaves": 7,
    "learning_rate": 0.05,
    "feature_fraction": 0.85,
    "bagging_fraction": 0.85,
    "bagging_freq": 1,
    "min_data_in_leaf": 5,
    "lambda_l1": 0.05,
    "lambda_l2": 0.20,
    "verbose": -1,
    "seed": 42,
    "n_jobs": -1,
}

LGBM_NUM_BOOST_ROUND = 120
LGBM_EARLY_STOPPING = 15
