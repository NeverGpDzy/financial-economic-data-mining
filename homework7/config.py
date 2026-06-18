"""Configuration for Homework 7 stationarity analysis."""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "homework7"
ORIGINAL_DIR = DATA_DIR / "original"
RAW_DIR = DATA_DIR / "raw"
OUTPUT_DIR = ROOT / "outputs" / "homework7"
REPORT_DIR = ROOT / "report"

STUDENT_NAME = "丁致宇"
STUDENT_ID = "202331060205"

START_DATE = "2024-01-01"
END_DATE = "2024-12-31"

STOCKS = {
    "中国石油": "sh.601857",
    "贵州茅台": "sh.600519",
    "兴蓉环境": "sz.000598",
    "招商银行": "sh.600036",
    "工商银行": "sh.601398",
}

ADF_P_THRESHOLD = 0.05

