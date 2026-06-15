"""Data loading and annual FCFF factor panel construction for Homework 6."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from . import config


def ensure_raw_data() -> None:
    """Extract the teacher-provided zip if the raw data folder is missing."""
    marker = config.RAW_ROOT / "merged_total.parquet"
    if marker.exists():
        return

    zip_path = config.ORIGINAL_DIR / "作业6 配套数据.zip"
    if not zip_path.exists():
        raise FileNotFoundError(f"缺少作业6配套数据压缩包: {zip_path}")

    config.DATA_DIR.joinpath("raw").mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(config.DATA_DIR / "raw")


def _parse_date(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series.astype(str), errors="coerce")


def _component_dates(components: pd.DataFrame) -> pd.DataFrame:
    comp = components.copy()
    comp["in_date_dt"] = pd.to_datetime(
        pd.to_numeric(comp["in_date"], errors="coerce").astype("Int64").astype(str),
        format="%Y%m%d",
        errors="coerce",
    )
    out_date = pd.to_numeric(comp["out_date"], errors="coerce")
    comp["out_date_dt"] = pd.to_datetime(
        out_date.where(out_date < 90000000).astype("Int64").astype(str).replace("<NA>", pd.NA),
        format="%Y%m%d",
        errors="coerce",
    )
    return comp[["ts_code", "in_date_dt", "out_date_dt"]]


def _mark_year_end_components(panel: pd.DataFrame, components: pd.DataFrame) -> pd.DataFrame:
    df = panel.copy()
    comp = _component_dates(components)
    df["rebalance_date"] = pd.to_datetime(df["year"].astype(str) + "-12-31")
    df["in_sz50_year_end"] = False
    for year in sorted(df["year"].dropna().unique()):
        year_end = pd.Timestamp(int(year), 12, 31)
        active = comp[
            (comp["in_date_dt"] <= year_end)
            & (comp["out_date_dt"].isna() | (comp["out_date_dt"] > year_end))
        ]["ts_code"]
        df.loc[df["year"].eq(year) & df["ts_code"].isin(active), "in_sz50_year_end"] = True
    return df


def load_raw_tables() -> dict[str, pd.DataFrame]:
    ensure_raw_data()
    root = config.RAW_ROOT
    tables = {
        "merged": pd.read_parquet(root / "merged_total.parquet"),
        "stock_daily": pd.read_parquet(root / "stock_daily.parquet"),
        "finance": pd.read_parquet(root / "finance_data.parquet"),
        "dividend": pd.read_parquet(root / "dividend_data.parquet"),
        "hs300": pd.read_csv(root / "hs300_index.csv"),
        "components": pd.read_csv(root / "sz50_dynamic_components.csv"),
    }
    return tables


def _safe_growth(current: pd.Series, future: pd.Series) -> pd.Series:
    base = current.abs()
    growth = (future - current) / base.replace(0, np.nan)
    return growth.replace([np.inf, -np.inf], np.nan)


def _annualized_fcff_growth(current: pd.Series, future: pd.Series, years: int) -> pd.Series:
    compound = pd.Series(np.nan, index=current.index, dtype=float)
    positive = (current > 0) & (future > 0)
    compound.loc[positive] = (future.loc[positive] / current.loc[positive]) ** (1 / years) - 1
    fallback = _safe_growth(current, future) / years
    compound = compound.fillna(fallback)
    return compound.replace([np.inf, -np.inf], np.nan)


def build_annual_panel(tables: dict[str, pd.DataFrame] | None = None) -> pd.DataFrame:
    """Build one row per stock-year with value factors and future FCFF labels."""
    if tables is None:
        tables = load_raw_tables()

    finance = tables["finance"].copy()
    finance["end_date_dt"] = _parse_date(finance["end_date"])
    finance["ann_date_dt"] = _parse_date(finance["ann_date"])
    finance = finance[
        finance["end_date_dt"].dt.month.eq(12) & finance["end_date_dt"].dt.day.eq(31)
    ].copy()
    finance["year"] = finance["end_date_dt"].dt.year
    finance = finance.sort_values(["ts_code", "year", "ann_date_dt"]).drop_duplicates(
        ["ts_code", "year"], keep="last"
    )

    merged = tables["merged"].copy()
    merged["trade_date"] = pd.to_datetime(merged["trade_date"])
    merged["year"] = merged["trade_date"].dt.year
    market_cols = [
        "ts_code",
        "year",
        "trade_date",
        "adj_close",
        "close",
        "total_share",
        "total_mv",
        "circ_mv",
        "pe",
        "pb",
        "div_yield",
        "hs300_close",
        "hs300_pct_chg",
    ]
    year_end_market = (
        merged.sort_values("trade_date")
        .groupby(["ts_code", "year"], as_index=False)
        .tail(1)[market_cols]
    )

    dividend = tables["dividend"].copy()
    dividend["end_date_dt"] = _parse_date(dividend["end_date"])
    dividend["year"] = dividend["end_date_dt"].dt.year
    dividend_annual = (
        dividend.groupby(["ts_code", "year"], as_index=False)
        .agg(cash_div_tax=("cash_div_tax", "sum"), stk_div=("stk_div", "sum"))
    )

    panel = finance.merge(year_end_market, on=["ts_code", "year"], how="left")
    panel = panel.merge(dividend_annual, on=["ts_code", "year"], how="left", suffixes=("", "_div"))
    if "cash_div_tax_div" in panel.columns:
        panel["cash_div_tax"] = panel["cash_div_tax_div"].fillna(panel.get("cash_div_tax", 0))
        panel.drop(columns=["cash_div_tax_div"], inplace=True)
    if "stk_div_div" in panel.columns:
        panel["stk_div"] = panel["stk_div_div"].fillna(panel.get("stk_div", 0))
        panel.drop(columns=["stk_div_div"], inplace=True)

    panel = panel.sort_values(["ts_code", "year"]).reset_index(drop=True)
    share_count = panel["total_share"] * 10_000.0
    equity = panel["bps"] * share_count
    total_assets = equity * panel["assets_to_eqt"]
    net_profit = panel["profit_dedt"].fillna(panel["q_dtprofit"])
    ocf = panel["ocfps"] * share_count

    panel["net_profit"] = net_profit
    panel["operating_cash_flow"] = ocf
    panel["fixed_asset_delta"] = (
        panel.groupby("ts_code")["fixed_assets"].diff().clip(lower=0).fillna(0)
    )
    panel["fcff"] = panel["operating_cash_flow"] - panel["fixed_asset_delta"]

    lag3_profit = panel.groupby("ts_code")["net_profit"].shift(3)
    cagr = pd.Series(np.nan, index=panel.index, dtype=float)
    positive_profit = (panel["net_profit"] > 0) & (lag3_profit > 0)
    cagr.loc[positive_profit] = (panel.loc[positive_profit, "net_profit"] / lag3_profit[positive_profit]) ** (1 / 3) - 1
    panel["F1_profit_cagr"] = cagr.fillna(panel["netprofit_yoy"] / 100.0)

    panel["F2_gross_margin"] = panel["grossprofit_margin"]
    panel["F3_net_margin"] = panel["netprofit_margin"]
    panel["F4_light_asset"] = panel["fixed_assets"] / total_assets.replace(0, np.nan)
    panel["F5_low_debt"] = panel["debt_to_assets"]
    panel["F6_roe"] = panel["roe_avg"].fillna(panel["roe_yearly"]).fillna(panel["roe"])
    panel["F7_low_expense"] = (
        panel[["saleexp_to_gr", "adminexp_of_gr", "finaexp_of_gr"]].fillna(0).sum(axis=1)
    )
    panel["F8_ocf_profit"] = panel["ocf_to_profit"]
    panel["F9_dividend_payout"] = np.where(
        panel["eps"] > 0,
        panel["cash_div_tax"].fillna(0) / panel["eps"],
        0.0,
    )
    panel["F10_dividend_yield"] = panel["div_yield"]
    panel["F11_fcff_yield"] = panel["fcff"] / (panel["total_mv"] * 10_000.0).replace(0, np.nan)

    future_fcff_1y = panel.groupby("ts_code")["fcff"].shift(-1)
    future_fcff_3y = panel.groupby("ts_code")["fcff"].shift(-3)
    panel["future_fcff_1y"] = future_fcff_1y
    panel["future_fcff_3y"] = future_fcff_3y
    panel["fcff_growth_1y"] = _safe_growth(panel["fcff"], future_fcff_1y)
    panel["fcff_growth_3y_ann"] = _annualized_fcff_growth(panel["fcff"], future_fcff_3y, 3)
    panel["future_adj_close_1y"] = panel.groupby("ts_code")["adj_close"].shift(-1)
    panel["price_return_1y"] = panel["future_adj_close_1y"] / panel["adj_close"] - 1
    panel["history_year_count"] = panel.groupby("ts_code").cumcount() + 1

    for col in config.FACTOR_COLUMNS + [
        "fcff",
        "fcff_growth_1y",
        "fcff_growth_3y_ann",
        "price_return_1y",
    ]:
        panel[col] = pd.to_numeric(panel[col], errors="coerce").replace([np.inf, -np.inf], np.nan)

    pre_filter_rows = len(panel)
    panel = _mark_year_end_components(panel, tables["components"])
    panel = panel[panel["in_sz50_year_end"] & panel["adj_close"].notna()].copy()
    panel.attrs["pre_component_filter_rows"] = pre_filter_rows
    panel.attrs["post_component_filter_rows"] = len(panel)

    return panel


def load_price_data(tables: dict[str, pd.DataFrame] | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    if tables is None:
        tables = load_raw_tables()

    stock = tables["stock_daily"].copy()
    stock["trade_date"] = _parse_date(stock["trade_date"])
    stock = stock.sort_values(["ts_code", "trade_date"])
    stock["ret"] = stock.groupby("ts_code")["adj_close"].pct_change()
    stock["year"] = stock["trade_date"].dt.year

    hs300 = tables["hs300"].copy()
    hs300["trade_date"] = _parse_date(hs300["trade_date"])
    hs300["benchmark_ret"] = pd.to_numeric(hs300["pct_chg"], errors="coerce") / 100.0
    hs300["year"] = hs300["trade_date"].dt.year
    return stock, hs300


def data_audit(panel: pd.DataFrame, tables: dict[str, pd.DataFrame]) -> dict:
    merged = tables["merged"]
    trade_dates = pd.to_datetime(merged["trade_date"])
    audit = {
        "raw_data_dir": str(config.RAW_ROOT),
        "merged_rows": int(len(merged)),
        "panel_rows_before_component_filter": int(panel.attrs.get("pre_component_filter_rows", len(panel))),
        "stock_count": int(panel["ts_code"].nunique()),
        "panel_rows": int(len(panel)),
        "panel_year_min": int(panel["year"].min()),
        "panel_year_max": int(panel["year"].max()),
        "trade_date_min": str(trade_dates.min().date()),
        "trade_date_max": str(trade_dates.max().date()),
        "assignment_period_note": (
            "题目正文写2006-2025；教师提供文件实际覆盖2014-03-31至2024-12-31，"
            "本次结果按2014-2024可复现口径完成。"
        ),
        "component_filter_note": (
            "年度因子面板仅保留对应财年12月31日仍属于上证50动态成分且存在年末行情快照的股票。"
        ),
    }
    return audit


def save_data_description(audit: dict) -> None:
    text = f"""# 作业6数据说明

原始材料来自父目录 `作业六/`，已复制到 `data/homework6/original/`。

## 原始文件

- `项目6： 价值投资内涵价值与市场定价 20260605 V3.docx`：教师发布的作业要求。
- `作业6 配套数据.zip`：教师提供的数据包，已解压到 `data/homework6/raw/作业6 配套数据/data/`。

## 数据文件

- `merged_total.parquet`：合并后的日度行情、财务、分红和沪深300字段。
- `stock_daily.parquet`：上证50成分股日度行情和复权价格。
- `finance_data.parquet`：季度和年度财务指标。
- `dividend_data.parquet`：年度分红数据。
- `hs300_index.csv`：沪深300指数日度行情。
- `sz50_dynamic_components.csv` / `sz50_quarterly_raw.csv`：上证50成分股记录。

## 可复现口径

- 实际行情日期：{audit['trade_date_min']} 至 {audit['trade_date_max']}。
- 年度财务截面：{audit['panel_year_min']} 至 {audit['panel_year_max']}。
- 成分股筛选前年度行数：{audit['panel_rows_before_component_filter']}。
- 股票数：{audit['stock_count']}。
- 说明：{audit['assignment_period_note']}
- 成分股口径：{audit['component_filter_note']}
"""
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    (config.DATA_DIR / "数据说明.md").write_text(text, encoding="utf-8")


def save_panel(panel: pd.DataFrame, audit: dict) -> None:
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    panel.to_csv(config.OUTPUT_DIR / "annual_factor_panel.csv", index=False, encoding="utf-8-sig")
    (config.OUTPUT_DIR / "data_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
