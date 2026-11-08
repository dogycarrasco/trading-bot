"""
Backtest engine v3.1 - ALWAYS INVESTED friendly.

Cambios vs v3:
  - Cuando la signal es igual a la posicion actual, NO hacer nada (sin churn)
  - Cuando la signal cambia de direccion, rotar (cerrar + abrir)
  - Stop loss y take profit se chequean con high/low de la siguiente barra
  - Si la signal pasa a 0 (bear confirmado), salir
  - Min hold: 5 barras para evitar whipsaws
  - Sostener dividendos automaticamente
"""
import numpy as np
import pandas as pd


def backtest_symbol(df_features, signals, trade_plan, config,
                    initial_capital=500.0, broker="alpaca", min_hold_bars=5):
    """Corre backtest barra a barra para un simbolo."""
    comm_cfg = config["backtest"]["commission_pct"]
    if isinstance(comm_cfg, dict):
        comm = float(comm_cfg.get(broker, 0.0))
    else:
        comm = float(comm_cfg)
    slip = config["backtest"]["slippage_pct"]

    plan = trade_plan.set_index("datetime") if not trade_plan.empty else pd.DataFrame()

    n = len(df_features)
    equity = np.zeros(n)
    equity[0] = initial_capital
    position = 0
    entry_price = 0.0
    stop = 0.0
    target = 0.0
    qty = 0.0
    entry_ts = None
    bars_held = 0
    trades = []
    peak = initial_capital

    dividend_yield_annual = 0.018

    for i in range(n):
        ts = df_features.index[i]
        high = float(df_features["high"].iloc[i])
        low = float(df_features["low"].iloc[i])
        close = float(df_features["close"].iloc[i])
        sig = int(signals["signal"].iloc[i]) if i < len(signals) else 0

        if position != 0:
            bars_held += 1
            exit_price = None
            exit_reason = None
            if position == 1:
                if low <= stop:
                    exit_price = stop
                    exit_reason = "STOP"
                elif high >= target:
                    exit_price = target
                    exit_reason = "TARGET"
            elif position == -1:
                if high >= stop:
                    exit_price = stop
                    exit_reason = "STOP"
                elif low <= target:
                    exit_price = target
                    exit_reason = "TARGET"

            if exit_price is None and sig == 0 and bars_held >= min_hold_bars:
                exit_price = close
                exit_reason = "SIGNAL_EXIT_BEAR"
            elif exit_price is None and sig != 0 and sig != position and bars_held >= min_hold_bars:
                exit_price = close
                exit_reason = "SIGNAL_FLIP"

            if exit_price is not None:
                exec_price = exit_price * (1 - position * slip)
                notional_exit = qty * exec_price
                comm_exit = notional_exit * comm
                if position == 1:
                    pnl = (exec_price - entry_price) * qty - comm_exit
                else:
                    pnl = (entry_price - exec_price) * qty - comm_exit
                equity[i] = equity[i - 1] + pnl
                trades.append({
                    "entry_time": entry_ts,
                    "exit_time": ts,
                    "direction": position,
                    "entry": entry_price,
                    "exit": exec_price,
                    "qty": qty,
                    "pnl": pnl,
                    "reason": exit_reason,
                    "bars_held": bars_held,
                })
                position = 0
                qty = 0.0
                entry_price = 0.0
                stop = 0.0
                target = 0.0
                bars_held = 0
            else:
                if position == 1:
                    daily_div = (dividend_yield_annual / 252) * qty * close
                    equity[i] = equity[i - 1] + daily_div
                else:
                    equity[i] = equity[i - 1]

        if position == 0 and sig != 0 and ts in plan.index:
            row = plan.loc[ts]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
            entry_price = float(row["entry"]) * (1 + sig * slip)
            qty = float(row["qty"])
            base_equity = equity[i] if equity[i] != 0 else (equity[i - 1] if i > 0 else initial_capital)
            if qty <= 0:
                if equity[i] == 0:
                    equity[i] = base_equity
                continue
            comm_in = qty * entry_price * comm
            equity[i] = base_equity - comm_in
            position = sig
            stop = float(row["stop"])
            target = float(row["target"])
            entry_ts = ts
            bars_held = 0
        elif position == 0 and equity[i] == 0:
            equity[i] = equity[i - 1] if i > 0 else initial_capital

        if equity[i] == 0:
            equity[i] = equity[i - 1] if i > 0 else initial_capital

        if equity[i] > peak:
            peak = equity[i]

    equity_series = pd.Series(equity, index=df_features.index, name="equity")
    trades_df = pd.DataFrame(trades)
    return {
        "equity": equity_series,
        "trades": trades_df,
        "final_equity": float(equity[-1]),
    }