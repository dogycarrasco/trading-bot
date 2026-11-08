"""
Estrategias base. Cada una devuelve una senal: +1 (long), -1 (short), 0 (no trade).
"""
import numpy as np
import pandas as pd


def momentum_signal(df):
    """Trend following: long cuando EMA9 > EMA21 > EMA50 y RSI < 70."""
    long_cond = (df["ema_9"] > df["ema_21"]) & (df["ema_21"] > df["ema_50"]) & (df["rsi_14"] < 70)
    short_cond = (df["ema_9"] < df["ema_21"]) & (df["ema_21"] < df["ema_50"]) & (df["rsi_14"] > 30)
    sig = np.where(long_cond, 1, np.where(short_cond, -1, 0))
    return pd.Series(sig, index=df.index)


def always_invested_momentum(df):
    """Estrategia always-invested (Meb Faber style): LONG si precio > EMA200."""
    long_cond = df["close"] > df["ema_200"]
    sig = np.where(long_cond, 1, 0)
    return pd.Series(sig, index=df.index)


def sector_rotation_signal(df, spy_returns=None):
    """Sector rotation: invertir en sectores que superan al SPY."""
    long_cond = (df["close"] > df["ema_200"]) & (df["ema_9"] > df["ema_21"])
    if spy_returns is not None:
        br = spy_returns.reindex(df.index).fillna(0)
        rs = df["ret_20"] - br
        long_cond = long_cond & (rs > -0.005)
    sig = np.where(long_cond, 1, 0)
    return pd.Series(sig, index=df.index)


def mean_reversion_signal(df):
    """Mean reversion: long cuando RSI < 35 y %B < 0.10. Short cuando RSI > 65 y %B > 0.90."""
    long_cond = (df["rsi_14"] < 35) & (df["bb_pct_b"] < 0.10)
    short_cond = (df["rsi_14"] > 65) & (df["bb_pct_b"] > 0.90)
    sig = np.where(long_cond, 1, np.where(short_cond, -1, 0))
    return pd.Series(sig, index=df.index)


def breakout_signal(df, lookback=50):
    """Breakout: long cuando close rompe el maximo de las ultimas `lookback` barras."""
    hh = df["high"].rolling(lookback).max().shift(1)
    ll = df["low"].rolling(lookback).min().shift(1)
    long_cond = (df["close"] > hh) & (df["macd_hist"] > 0)
    short_cond = (df["close"] < ll) & (df["macd_hist"] < 0)
    sig = np.where(long_cond, 1, np.where(short_cond, -1, 0))
    return pd.Series(sig, index=df.index)


def trend_strength_signal(df):
    """Trend strength: solo entra cuando ADX > 30 (trend fuerte)."""
    strong_trend = df["adx_14"] > 30
    long_cond = strong_trend & (df["close"] > df["ema_21"]) & (df["rsi_14"] > 40) & (df["rsi_14"] < 65)
    short_cond = strong_trend & (df["close"] < df["ema_21"]) & (df["rsi_14"] < 60) & (df["rsi_14"] > 35)
    sig = np.where(long_cond, 1, np.where(short_cond, -1, 0))
    return pd.Series(sig, index=df.index)


def relative_strength_signal(df, benchmark_ret=None):
    """Relative strength vs SPY (benchmark)."""
    if benchmark_ret is None:
        long_cond = (df["close"] > df["ema_21"]) & (df["ret_20"] > 0)
        sig = np.where(long_cond, 1, 0)
        return pd.Series(sig, index=df.index)
    br = benchmark_ret.reindex(df.index).fillna(0)
    rs = df["ret_20"] - br
    long_cond = rs > 0.01
    short_cond = rs < -0.02
    sig = np.where(long_cond, 1, np.where(short_cond, -1, 0))
    return pd.Series(sig, index=df.index)


def dividend_capture_signal(df):
    """Mantener posicion en ETFs de alto dividendo cuando hay tendencia alcista de fondo."""
    long_cond = (df["rsi_14"] < 50) & (df["close"] > df["ema_200"]) & (df["vol_20"] < df["vol_50"])
    sig = np.where(long_cond, 1, 0)
    return pd.Series(sig, index=df.index)


def ml_signal_wrapper(ml_probs, threshold=0.50):
    """Wrapper que convierte probs del ML en senal."""
    sig = np.where(ml_probs["prob_up"] > threshold, 1,
                   np.where(ml_probs["prob_down"] > threshold, -1, 0))
    return pd.Series(sig, index=ml_probs.index)