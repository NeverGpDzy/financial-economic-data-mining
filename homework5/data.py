"""Data loading and market benchmark fetching for Homework 5."""

from __future__ import annotations

import json
import shutil
import zipfile
from datetime import datetime
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

from . import config


def ensure_stock_price_csv(path: Path = config.STOCK_PRICE_CSV) -> Path:
    """Ensure the raw stock CSV exists, extracting it from the original zip if needed."""
    if path.exists():
        return path

    zip_path = config.STOCK_PRICE_ZIP
    if not zip_path.exists():
        raise FileNotFoundError(f"未找到股票行情CSV，也未找到原始压缩包: {path} / {zip_path}")

    config.RAW_DIR.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        csv_members = [name for name in zf.namelist() if name.lower().endswith(".csv")]
        if not csv_members:
            raise FileNotFoundError(f"压缩包中没有CSV文件: {zip_path}")

        member = csv_members[0]
        target = path
        with zf.open(member) as src, target.open("wb") as dst:
            shutil.copyfileobj(src, dst)

    return path


def load_stock_prices(path: Path = config.STOCK_PRICE_CSV) -> pd.DataFrame:
    """Load teacher-provided SSE 50 constituent daily prices."""
    path = ensure_stock_price_csv(path)
    if not path.exists():
        raise FileNotFoundError(f"未找到股票行情CSV: {path}")

    df = pd.read_csv(path, parse_dates=["date"])
    required = {"date", "code", "open", "high", "low", "close", "volume", "amount"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"股票行情CSV缺少字段: {sorted(missing)}")

    for col in ["open", "high", "low", "close", "volume", "amount"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = (
        df.dropna(subset=["date", "code", "close"])
        .drop_duplicates(subset=["date", "code"])
        .sort_values(["code", "date"])
        .reset_index(drop=True)
    )
    return df


def add_stock_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Compute daily stock returns from close prices."""
    out = prices.copy()
    out["stock_return"] = out.groupby("code", group_keys=False)["close"].pct_change()
    out = out.replace([np.inf, -np.inf], np.nan).dropna(subset=["stock_return"])
    return out


def fetch_market_eastmoney(
    start: str = config.TRAIN_START,
    end: str = config.PRIMARY_TEST_END,
) -> pd.DataFrame:
    """Fetch CSI 300 daily K-line from Eastmoney public endpoint."""
    params = {
        "secid": config.MARKET_EASTMONEY_SECID,
        "fields1": "f1,f2,f3,f4,f5,f6",
        "fields2": "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61",
        "klt": "101",
        "fqt": "1",
        "beg": start.replace("-", ""),
        "end": end.replace("-", ""),
    }
    url = "https://push2his.eastmoney.com/api/qt/stock/kline/get?" + urlencode(params)
    request = Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0",
            "Referer": "https://quote.eastmoney.com/",
        },
    )
    with urlopen(request, timeout=30) as response:
        payload = json.loads(response.read().decode("utf-8"))

    if payload.get("rc") != 0 or not payload.get("data") or not payload["data"].get("klines"):
        raise RuntimeError(f"东方财富沪深300数据获取失败: {payload}")

    rows = []
    for line in payload["data"]["klines"]:
        parts = line.split(",")
        rows.append(
            {
                "date": parts[0],
                "open": parts[1],
                "close": parts[2],
                "high": parts[3],
                "low": parts[4],
                "volume": parts[5],
                "amount": parts[6],
            }
        )

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    for col in ["open", "close", "high", "low", "volume", "amount"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.dropna(subset=["date", "close"]).sort_values("date").reset_index(drop=True)


def fetch_market_baostock(
    start: str = config.TRAIN_START,
    end: str = config.PRIMARY_TEST_END,
) -> pd.DataFrame:
    """Fetch CSI 300 close prices from Baostock when available."""
    import baostock as bs

    login = bs.login()
    if login.error_code != "0":
        raise RuntimeError(f"baostock登录失败: {login.error_msg}")

    try:
        rs = bs.query_history_k_data_plus(
            config.MARKET_BAOSTOCK_CODE,
            "date,open,high,low,close,volume,amount",
            start_date=start,
            end_date=end,
            frequency="d",
            adjustflag="3",
        )
        if rs.error_code != "0":
            raise RuntimeError(f"baostock查询失败: {rs.error_msg}")

        rows = []
        while rs.next():
            rows.append(rs.get_row_data())
        df = pd.DataFrame(rows, columns=rs.fields)
    finally:
        bs.logout()

    if df.empty:
        raise RuntimeError("baostock未返回沪深300数据")

    df["date"] = pd.to_datetime(df["date"])
    for col in ["open", "high", "low", "close", "volume", "amount"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.dropna(subset=["date", "close"]).sort_values("date").reset_index(drop=True)


def build_equal_weight_market_proxy(stock_returns: pd.DataFrame) -> pd.DataFrame:
    """Build an offline market-return proxy from the teacher-provided stock pool."""
    proxy = (
        stock_returns.groupby("date", as_index=False)["stock_return"]
        .mean()
        .rename(columns={"stock_return": "market_return"})
        .sort_values("date")
    )
    proxy["close"] = 1000.0 * (1.0 + proxy["market_return"]).cumprod()
    return proxy[["date", "close", "market_return"]]


def add_market_returns(market: pd.DataFrame) -> pd.DataFrame:
    """Add market returns if not already present."""
    out = market.copy().sort_values("date")
    if "market_return" not in out.columns:
        out["market_return"] = out["close"].pct_change()
    out = out.replace([np.inf, -np.inf], np.nan).dropna(subset=["market_return"])
    return out[["date", "close", "market_return"]]


def _write_market_cache(market: pd.DataFrame, source_name: str) -> None:
    config.RAW_DIR.mkdir(parents=True, exist_ok=True)
    market.to_csv(config.MARKET_CACHE_CSV, index=False, encoding="utf-8-sig")
    meta = {
        "source": source_name,
        "market": config.MARKET_NAME,
        "date_min": str(market["date"].min().date()),
        "date_max": str(market["date"].max().date()),
        "rows": int(len(market)),
        "cached_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    config.MARKET_META_JSON.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


def _read_market_source_from_meta() -> str:
    if config.MARKET_META_JSON.exists():
        try:
            meta = json.loads(config.MARKET_META_JSON.read_text(encoding="utf-8"))
            source = meta.get("source")
            if source:
                return f"{source}（本地缓存）"
        except json.JSONDecodeError:
            pass
    return f"{config.MARKET_NAME}本地缓存"


def _cache_matches_requested_source(source: str) -> bool:
    if source == "auto":
        return True
    if not config.MARKET_META_JSON.exists():
        return False
    try:
        meta = json.loads(config.MARKET_META_JSON.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    cached_source = str(meta.get("source", "")).lower()
    if source == "eastmoney":
        return "东方财富" in cached_source or "eastmoney" in cached_source
    if source == "baostock":
        return "baostock" in cached_source
    return False


def load_or_fetch_market(
    stock_returns: pd.DataFrame,
    refresh: bool = False,
    source: str = "auto",
) -> tuple[pd.DataFrame, str]:
    """Load cached CSI 300 data or fetch it online, with an offline proxy fallback."""
    config.RAW_DIR.mkdir(parents=True, exist_ok=True)
    cache = config.MARKET_CACHE_CSV

    if (
        cache.exists()
        and not refresh
        and source in {"auto", "eastmoney", "baostock"}
        and _cache_matches_requested_source(source)
    ):
        market = pd.read_csv(cache, parse_dates=["date"])
        return add_market_returns(market), _read_market_source_from_meta()

    errors: list[str] = []
    if source in {"auto", "eastmoney"}:
        try:
            source_name = "东方财富沪深300公开K线"
            market = fetch_market_eastmoney()
            _write_market_cache(market, source_name)
            return add_market_returns(market), source_name
        except Exception as exc:
            errors.append(f"eastmoney: {exc}")

    if source in {"auto", "baostock"}:
        try:
            source_name = "baostock沪深300"
            market = fetch_market_baostock()
            _write_market_cache(market, source_name)
            return add_market_returns(market), source_name
        except Exception as exc:
            errors.append(f"baostock: {exc}")

    if source in {"auto", "proxy"}:
        return build_equal_weight_market_proxy(stock_returns), "上证50成分股等权收益代理"

    raise RuntimeError("无法获取市场基准数据: " + " | ".join(errors))


def data_audit(stock_prices: pd.DataFrame, market: pd.DataFrame, market_source: str) -> dict:
    """Summarize loaded data for reports."""
    counts = stock_prices.groupby("code")["date"].agg(["min", "max", "count"])
    return {
        "stock_rows": int(len(stock_prices)),
        "stock_count": int(stock_prices["code"].nunique()),
        "stock_date_min": str(stock_prices["date"].min().date()),
        "stock_date_max": str(stock_prices["date"].max().date()),
        "stock_obs_min": int(counts["count"].min()),
        "stock_obs_median": float(counts["count"].median()),
        "stock_obs_max": int(counts["count"].max()),
        "market_source": market_source,
        "market_rows": int(len(market)),
        "market_date_min": str(market["date"].min().date()),
        "market_date_max": str(market["date"].max().date()),
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
