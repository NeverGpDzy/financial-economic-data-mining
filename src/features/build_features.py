"""构建滚动时序特征。"""

import pandas as pd


def build_features(df: pd.DataFrame, window: int = 10) -> pd.DataFrame:
    """计算价格收益率、成交量收益率，并构造滚动窗口特征。

    特征：过去 window 天的价格收益率 + 成交量收益率（共 2*window 个特征）
    标签：次日价格收益率

    Args:
        df: 必须包含 close, volume 列，index 为日期
        window: 滚动窗口天数，默认 10

    Returns:
        包含特征和标签的 DataFrame，已删除缺失值
    """
    df = df.copy()

    # 日价格收益率
    df["price_return"] = df["close"].pct_change()
    # 日成交量收益率
    df["vol_return"] = df["volume"].pct_change()
    # 标签：次日价格收益率
    df["label"] = df["price_return"].shift(-1)

    # 滚动窗口特征
    for i in range(1, window + 1):
        df[f"price_return_{i}"] = df["price_return"].shift(i)
        df[f"vol_return_{i}"] = df["vol_return"].shift(i)

    df = df.dropna()

    feature_cols = (
        [f"price_return_{i}" for i in range(1, window + 1)]
        + [f"vol_return_{i}" for i in range(1, window + 1)]
    )

    return df, feature_cols
