"""
Feature engineering for the ML model.

Indicators are hand-rolled rather than pulled from a technical-analysis
library on purpose: every formula here is auditable in about one line, so
if a feature looks wrong in production you can check the math yourself
instead of debugging someone else's package.

All functions take a DataFrame with columns [open, high, low, close, volume]
indexed by timestamp, and return a Series or DataFrame aligned to that
same index. Every feature here only uses data up to and including bar t --
never bar t+1 or later. That's the #1 way amateur ML trading bots produce
backtest numbers that evaporate live: a feature that accidentally peeks
at the future (e.g. normalizing by full-series min/max, or using a
centered rolling window) makes the model look prescient on history and
useless in production.
"""

from __future__ import annotations
import numpy as np
import pandas as pd


def returns(close: pd.Series, periods: int = 1) -> pd.Series:
    return close.pct_change(periods)


def log_returns(close: pd.Series, periods: int = 1) -> pd.Series:
    return np.log(close / close.shift(periods))


def sma(close: pd.Series, window: int) -> pd.Series:
    return close.rolling(window, min_periods=window).mean()


def ema(close: pd.Series, span: int) -> pd.Series:
    return close.ewm(span=span, adjust=False, min_periods=span).mean()


def rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    ema_fast = ema(close, fast)
    ema_slow = ema(close, slow)
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    hist = macd_line - signal_line
    return pd.DataFrame({"macd": macd_line, "macd_signal": signal_line, "macd_hist": hist})


def bollinger_pct_b(close: pd.Series, window: int = 20, n_std: float = 2.0) -> pd.Series:
    mid = sma(close, window)
    std = close.rolling(window, min_periods=window).std()
    upper = mid + n_std * std
    lower = mid - n_std * std
    # %B: where price sits within the bands. 0 = at lower band, 1 = at upper.
    return (close - lower) / (upper - lower).replace(0, np.nan)


def atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()


def realized_vol(close: pd.Series, window: int = 20) -> pd.Series:
    return log_returns(close).rolling(window, min_periods=window).std()


def volume_zscore(volume: pd.Series, window: int = 20) -> pd.Series:
    mean = volume.rolling(window, min_periods=window).mean()
    std = volume.rolling(window, min_periods=window).std()
    return (volume - mean) / std.replace(0, np.nan)


def session_features(index: pd.DatetimeIndex) -> pd.DataFrame:
    """
    Time-of-day / day-of-week dummies. Matters more for forex (London/NY/
    Tokyo session overlap drives liquidity and volatility) than crypto,
    but harmless either way. Uses UTC hour, so make sure the incoming
    index is UTC (all providers in this platform return UTC-indexed data).
    """
    hour = index.hour + index.minute / 60.0
    return pd.DataFrame(
        {
            "hour_sin": np.sin(2 * np.pi * hour / 24),
            "hour_cos": np.cos(2 * np.pi * hour / 24),
            "dow_sin": np.sin(2 * np.pi * index.dayofweek / 7),
            "dow_cos": np.cos(2 * np.pi * index.dayofweek / 7),
        },
        index=index,
    )


def build_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Assemble the full feature set the model trains on. `df` must have
    columns [open, high, low, close, volume]; any extra columns (e.g. the
    synthetic provider's diagnostic `regime` column) are ignored.

    Returns a DataFrame of features only (no OHLCV, no target) aligned to
    the same index as `df`, with leading NaN rows wherever a rolling
    window hasn't filled yet -- drop those before training.
    """
    close, high, low, volume = df["close"], df["high"], df["low"], df["volume"]

    feats = pd.DataFrame(index=df.index)
    for p in (1, 3, 5, 10, 20):
        feats[f"logret_{p}"] = log_returns(close, p)

    feats["sma_10_ratio"] = close / sma(close, 10) - 1
    feats["sma_50_ratio"] = close / sma(close, 50) - 1
    feats["ema_20_ratio"] = close / ema(close, 20) - 1

    feats["rsi_14"] = rsi(close, 14) / 100.0  # scale to ~[0,1]

    macd_df = macd(close)
    feats["macd_hist_norm"] = macd_df["macd_hist"] / close

    feats["bb_pct_b"] = bollinger_pct_b(close)

    atr_14 = atr(high, low, close, 14)
    feats["atr_pct"] = atr_14 / close  # normalized so it's comparable across price levels

    feats["realized_vol_20"] = realized_vol(close, 20)
    feats["vol_zscore_20"] = volume_zscore(volume, 20)

    feats = feats.join(session_features(df.index))

    return feats
