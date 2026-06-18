"""Shared paths and parameters for Experiment 2."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKAGE_DIR = ROOT / "experiment2"
DATA_DIR = ROOT / "data" / "experiment2"
RAW_DIR = DATA_DIR / "raw"
OUTPUT_DIR = ROOT / "outputs" / "experiment2"
REPORT_DIR = ROOT / "report"

INPUT_WEEKLY_SENTIMENT = ROOT / "outputs" / "experiment1" / "weekly_sentiment.csv"
DB_FILE = OUTPUT_DIR / "experiment2.db"

ROLLING_BASELINE_WEEKS = 4
OUTLIER_SIGMA = 3.0

STUDENT_ID = "202331060205"
STUDENT_NAME = "丁致宇"

