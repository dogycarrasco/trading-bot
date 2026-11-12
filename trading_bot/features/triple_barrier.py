"""
Triple Barrier Labeling - Lopez de Prado (AFML Chapter 3).
"""
import os
import sys
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)


def compute_atr(df, period=14):
    if df is None or df.empty or len(df) < period + 1:
        return None
    high = df["high"]
    low = df["low"]
    close = df["close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low),
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1/period, adjust=False).mean()


def atr_at_date(df, date, period=14):
    if df is None or df.empty:
        return None
    try:
        sub = df.loc[:date]
        if len(sub) < period + 1:
            return None
        atr = compute_atr(sub, period)
        if atr is None or atr.empty:
            return None
        return float(atr.iloc[-1])
    except Exception:
        return None


def compute_barriers(entry_price, atr_value, pt_factor=5.0, sl_factor=5.0,
                     max_hold_days=63):
    if atr_value is None or atr_value <= 0 or entry_price <= 0:
        return {
            "tp_price": entry_price * 1.10,
            "sl_price": entry_price * 0.95,
            "max_hold": max_hold_days,
            "atr": atr_value,
            "tp_pct": 0.10,
            "sl_pct": 0.05,
        }
    tp_price = entry_price + pt_factor * atr_value
    sl_price = entry_price - sl_factor * atr_value
    return {
        "tp_price": tp_price,
        "sl_price": sl_price,
        "max_hold": max_hold_days,
        "atr": atr_value,
        "tp_pct": (tp_price / entry_price - 1),
        "sl_pct": (sl_price / entry_price - 1),
    }


def evaluate_trade_outcome(df, entry_date, entry_price, barriers,
                           max_hold_days=None):
    if df is None or df.empty or entry_date not in df.index:
        return None
    if max_hold_days is None:
        max_hold_days = barriers.get("max_hold", 63)
    tp = barriers["tp_price"]
    sl = barriers["sl_price"]
    entry_idx = df.index.get_loc(entry_date)
    future = df.iloc[entry_idx + 1: entry_idx + 1 + max_hold_days]
    if future.empty:
        return {"outcome": 0, "exit_date": entry_date, "exit_price": entry_price, "pnl_pct": 0.0, "hold_days": 0}
    exit_date = future.index[-1]
    exit_price = float(future["close"].iloc[-1])
    outcome = 0
    for idx, row in future.iterrows():
        high = float(row["high"])
        low = float(row["low"])
        if low <= sl:
            outcome = -1
            exit_date = idx
            exit_price = sl
            break
        if high >= tp:
            outcome = +1
            exit_date = idx
            exit_price = tp
            break
    else:
        outcome = 0
        exit_date = future.index[-1]
        exit_price = float(future["close"].iloc[-1])
    pnl_pct = (exit_price / entry_price - 1) if entry_price > 0 else 0
    hold_days = (pd.Timestamp(exit_date) - pd.Timestamp(entry_date)).days
    return {"outcome": outcome, "exit_date": exit_date, "exit_price": exit_price, "pnl_pct": pnl_pct, "hold_days": hold_days}


def build_label_for_meta_model(outcome):
    if outcome["outcome"] == +1:
        return 1
    elif outcome["outcome"] == -1:
        return 0
    else:
        return 1 if outcome["pnl_pct"] > 0 else 0
