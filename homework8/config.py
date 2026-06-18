"""Configuration for Homework 8 cointegration and pair trading."""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "homework8"
ORIGINAL_DIR = DATA_DIR / "original"
RAW_DIR = DATA_DIR / "raw"
OUTPUT_DIR = ROOT / "outputs" / "homework8"
REPORT_DIR = ROOT / "report"

STUDENT_NAME = "丁致宇"
STUDENT_ID = "202331060205"

START_DATE = "2015-01-01"
END_DATE = "2018-01-01"

# The assignment's basic information lists eight targets. The later prompt
# omits Luzhou Laojiao and Changjiang Power, so the full pool is retained here.
ASSETS = {
    "中国石油": "sh.601857",
    "贵州茅台": "sh.600519",
    "泸州老窖": "sz.000568",
    "兴蓉环境": "sz.000598",
    "招商银行": "sh.600036",
    "工商银行": "sh.601398",
    "长江电力": "sh.600900",
    "上证50指数": "sh.000016",
}

ADF_P_THRESHOLD = 0.05
ZSCORE_ENTRY = 2.0
ZSCORE_EXIT = 0.0

