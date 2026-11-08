"""
Risk manager adaptativo.
"""
import numpy as np
import pandas as pd


def compute_size_pct(regime, mode, config, broker=""):
    base = {
        "TREND_UP": config["risk"]["base_risk_per_trade"] * 1.3,
        "TREND_DOWN": config["risk"]["base_risk_per_trade"] * 1.0,
        "RANGE": config["risk"]["base_risk_per_trade"] * 0.75,
        "HIGH_VOL": config["risk"]["base_risk_per_trade"] * 0.5,
    }.get(regime, config["risk"]["base_risk_per_trade"] * 0.5)

    if mode == "RECOVERY":
        base *= 0.5
    if mode == "CAPITAL_PRESERVATION":
        return 0.0

    base = max(config["risk"]["min_risk_per_trade"], min(config["risk"]["max_risk_per_trade"], base))
    return base


def build_trade_plan(df_features, signals, equity, config, broker=""):
    """
    Por cada barra donde signal != 0, calcula size, stop, target.
    Aplica apalancamiento cripto 2x si broker es binance.
    """
    stop_mult = config["risk"]["stop_loss_atr_mult"]
    tp_mult = config["risk"]["take_profit_atr_mult"]
    leverage = 1.0
    if "binance" in broker.lower():
        leverage = config["backtest"].get("cripto_leverage", 1.0)

    rows = []
    for i in range(len(df_features)):
        sig = int(signals["signal"].iloc[i])
        if sig == 0:
            continue
        row_ts = df_features.index[i]
        close = float(df_features["close"].iloc[i])
        atr = float(df_features["atr_14"].iloc[i])
        if atr <= 0 or np.isnan(atr):
            continue

        regime = signals["regime"].iloc[i]
        mode = signals["mode"].iloc[i]
        size_pct = compute_size_pct(regime, mode, config, broker)
        if size_pct <= 0:
            continue

        eq = float(equity.iloc[i]) if i < len(equity) else float(equity.iloc[-1])
        eff_leverage = leverage if regime in ("TREND_UP", "TREND_DOWN") else 1.0
        risk_dollars = eq * size_pct * eff_leverage

        stop = close - sig * stop_mult * atr
        target = close + sig * tp_mult * atr

        risk_per_unit = abs(close - stop)
        if risk_per_unit <= 0:
            continue
        position_dollars = risk_dollars / risk_per_unit * close
        max_notional = eq * eff_leverage * 0.95
        position_dollars = min(position_dollars, max_notional)
        qty = position_dollars / close

        rows.append({
            "datetime": row_ts,
            "signal": sig,
            "regime": regime,
            "mode": mode,
            "strategy": signals["strategy_used"].iloc[i],
            "entry": close,
            "stop": stop,
            "target": target,
            "atr": atr,
            "size_pct": size_pct,
            "leverage": eff_leverage,
            "qty": qty,
            "notional": qty * close,
            "risk_dollars": qty * risk_per_unit,
        })

    return pd.DataFrame(rows)