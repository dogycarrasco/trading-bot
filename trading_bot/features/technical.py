"""
Features tecnicos calculados sobre series OHLCV.
Todos los features son funciones puras que reciben un DataFrame OHLCV y devuelven
un DataFrame con las mismas filas + columnas nuevas. No hay lookahead.
"""
import numpy as np
import pandas as pd


def rsi(close, period=14):
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(50)


def ema(series, period):
    return series.ewm(span=period, adjust=False).mean()


def atr(df, period=14):
    hl = df["high"] - df["low"]
    hc = (df["high"] - df["close"].shift()).abs()
    lc = (df["low"] - df["close"].shift()).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


def bollinger(close, period=20, n_std=2.0):
    mid = close.rolling(period).mean()
    std = close.rolling(period).std()
    upper = mid + n_std * std
    lower = mid - n_std * std
    pct_b = (close - lower) / (upper - lower).replace(0, np.nan)
    return mid, upper, lower, pct_b


def macd(close, fast=12, slow=26, signal=9):
    ema_fast = ema(close, fast)
    ema_slow = ema(close, slow)
    macd_line = ema_fast - ema_slow
    sig = ema(macd_line, signal)
    hist = macd_line - sig
    return macd_line, sig, hist


def adx(df, period=14):
    up = df["high"].diff()
    down = -df["low"].diff()
    plus_dm = np.where((up > down) & (up > 0), up, 0.0)
    minus_dm = np.where((down > up) & (down > 0), down, 0.0)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - df["close"].shift()).abs(),
        (df["low"] - df["close"].shift()).abs(),
    ], axis=1).max(axis=1)
    atr_ = tr.ewm(alpha=1 / period, adjust=False).mean()
    plus_di = 100 * pd.Series(plus_dm, index=df.index).ewm(alpha=1 / period, adjust=False).mean() / atr_
    minus_di = 100 * pd.Series(minus_dm, index=df.index).ewm(alpha=1 / period, adjust=False).mean() / atr_
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return dx.ewm(alpha=1 / period, adjust=False).mean().fillna(0)


def returns(close, horizons=(1, 3, 5, 10, 20)):
    out = {}
    for h in horizons:
        out[f"ret_{h}"] = close.pct_change(h)
    return pd.DataFrame(out)


def volatility(close, window=20):
    return close.pct_change().rolling(window).std() * np.sqrt(window)


def add_all_features(df):
    """Anade todos los features a un OHLCV df. Devuelve copia."""
    out = df.copy()
    close, high, low = out["close"], out["high"], out["low"]

    out["ema_9"] = ema(close, 9)
    out["ema_21"] = ema(close, 21)
    out["ema_50"] = ema(close, 50)
    out["ema_200"] = ema(close, 200)

    out["rsi_14"] = rsi(close, 14)
    out["rsi_7"] = rsi(close, 7)

    out["atr_14"] = atr(out, 14)
    out["atr_pct"] = out["atr_14"] / close

    mid, up, lo, pct_b = bollinger(close, 20, 2.0)
    out["bb_mid"] = mid
    out["bb_upper"] = up
    out["bb_lower"] = lo
    out["bb_pct_b"] = pct_b

    m_line, m_sig, m_hist = macd(close)
    out["macd_line"] = m_line
    out["macd_signal"] = m_sig
    out["macd_hist"] = m_hist

    out["adx_14"] = adx(out, 14)

    rets = returns(close)
    for c in rets.columns:
        out[c] = rets[c]

    out["vol_20"] = volatility(close, 20)
    out["vol_50"] = volatility(close, 50)

    out["price_above_ema50"] = (close > out["ema_50"]).astype(int)
    out["price_above_ema200"] = (close > out["ema_200"]).astype(int)
    out["ema9_above_ema21"] = (out["ema_9"] > out["ema_21"]).astype(int)
    out["macd_hist_pos"] = (out["macd_hist"] > 0).astype(int)

    return out