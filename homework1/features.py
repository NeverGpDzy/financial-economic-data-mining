"""作业1：滚动时序特征构建。"""

import pandas as pd


def build_features(df: pd.DataFrame, window: int = 10) -> tuple[pd.DataFrame, list[str]]:
    """计算价格收益率、成交量收益率，构造滚动窗口特征。

    特征：过去 window 天的价格收益率 + 成交量收益率
    标签：次日价格收益率
    """
    df = df.copy()

    df["price_return"] = df["close"].pct_change()
    df["vol_return"] = df["volume"].pct_change()
    df["label"] = df["price_return"].shift(-1)

    for i in range(1, window + 1):
        df[f"price_return_{i}"] = df["price_return"].shift(i)
        df[f"vol_return_{i}"] = df["vol_return"].shift(i)

    df = df.dropna()

    feature_cols = (
        [f"price_return_{i}" for i in range(1, window + 1)]
        + [f"vol_return_{i}" for i in range(1, window + 1)]
    )

    return df, feature_cols
