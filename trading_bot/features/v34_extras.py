"""
v34 extras: dividendos, boom-crash detector, MTF inteligente.
"""
import os
import sys
import warnings
import numpy as np
import pandas as pd
from collections import deque

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

ETF_DIVIDENDS = {
    "SPY": 1.5, "QQQ": 0.6, "XLK": 0.7, "XLF": 1.8,
    "XLE": 3.5, "XLV": 1.4, "XLY": 0.8, "XLP": 2.5,
    "XLI": 1.3, "XLU": 3.2, "GLD": 0.0, "USO": 0.0,
    "BTCUSD": 0.0, "ETHUSD": 0.0, "SOLUSD": 0.0,
    "BNBUSD": 0.0, "XRPUSD": 0.0, "ADAUSD": 0.0,
}


class BoomCrashDetector:
    def __init__(self):
        self.history = deque(maxlen=20)

    def check(self, df, date):
        if df is None or df.empty:
            return False
        ts = pd.Timestamp(date)
        try:
            sub = df.loc[:ts]
            if len(sub) < 10:
                return False
            close = sub["close"]
            last_ret = (close.iloc[-1] / close.iloc[-2] - 1) * 100
            vol = close.pct_change().tail(20).std() * 100
            is_boom = last_ret > vol * 3 and last_ret > 5
            if len(close) >= 3:
                prev_ret = (close.iloc[-2] / close.iloc[-3] - 1) * 100
                if prev_ret > vol * 3 and last_ret < 0:
                    return True
            return is_boom
        except Exception:
            return False


def check_daily_weekly_alignment(df, date):
    if df is None or df.empty:
        return True, 1.0
    ts = pd.Timestamp(date)
    sub = df.loc[:ts]
    if len(sub) < 50:
        return True, 1.0
    close = sub["close"]
    ema20_d = close.ewm(span=20, adjust=False).mean().iloc[-1]
    ema50_d = close.ewm(span=50, adjust=False).mean().iloc[-1]
    daily_bull = ema20_d > ema50_d
    try:
        weekly = close.resample("W-FRI").last().dropna()
        if len(weekly) >= 20:
            ema10_w = weekly.ewm(span=10, adjust=False).mean().iloc[-1]
            ema20_w = weekly.ewm(span=20, adjust=False).mean().iloc[-1]
            weekly_bull = ema10_w > ema20_w
        else:
            weekly_bull = daily_bull
    except Exception:
        weekly_bull = daily_bull
    aligned = (daily_bull and weekly_bull) or (not daily_bull and not weekly_bull)
    return (True, 1.0) if aligned else (False, 0.5)
