"""
Metricas de backtest.
Maneja correctamente distintos timeframes (1d, 4h, 1h) para anualizar.
"""
import numpy as np
import pandas as pd


def compute_metrics(equity, trades, initial_capital=500.0, timeframe="1d"):
    """Calcula metricas estandar de un backtest."""
    if equity.empty:
        return {}

    if timeframe in ("1d", "1D"):
        periods_per_year = 252
        bars_per_day = 1
    elif timeframe in ("4h", "4H"):
        periods_per_year = 365 * 6
        bars_per_day = 6
    elif timeframe in ("1h", "60m"):
        periods_per_year = 365 * 24
        bars_per_day = 24
    else:
        periods_per_year = 252
        bars_per_day = 1

    rets = equity.pct_change().dropna()
    rets_active = rets[rets != 0]
    final = float(equity.iloc[-1])
    total_return = (final / initial_capital) - 1

    n_bars = len(equity)
    days = n_bars / bars_per_day if bars_per_day > 0 else n_bars
    years = days / 365 if days > 0 else 1
    cagr = (final / initial_capital) ** (1 / years) - 1 if years > 0 and final > 0 else 0

    if rets.std() > 0:
        sharpe = float(rets.mean() / rets.std() * np.sqrt(periods_per_year))
    else:
        sharpe = 0.0

    if len(rets_active) > 5 and rets_active.std() > 0:
        sharpe_active = float(rets_active.mean() / rets_active.std() * np.sqrt(periods_per_year))
    else:
        sharpe_active = 0.0

    downside = rets[rets < 0]
    if len(downside) > 0 and downside.std() > 0:
        sortino = float(rets.mean() / downside.std() * np.sqrt(periods_per_year))
    else:
        sortino = 0.0 if rets.std() == 0 else float("inf")

    peak = equity.cummax()
    dd = (equity - peak) / peak
    max_dd = float(dd.min())
    max_dd_date = str(dd.idxmin()) if not dd.empty else ""

    vol = float(rets.std() * np.sqrt(periods_per_year)) if rets.std() > 0 else 0.0

    if not trades.empty:
        wins = trades[trades["pnl"] > 0]
        losses = trades[trades["pnl"] <= 0]
        n_trades = len(trades)
        win_rate = len(wins) / n_trades if n_trades > 0 else 0
        avg_win = float(wins["pnl"].mean()) if len(wins) > 0 else 0
        avg_loss = float(losses["pnl"].mean()) if len(losses) > 0 else 0
        gross_profit = float(wins["pnl"].sum()) if len(wins) > 0 else 0
        gross_loss = float(losses["pnl"].sum()) if len(losses) > 0 else 0
        profit_factor = (gross_profit / abs(gross_loss)) if gross_loss != 0 else float("inf")
        expectancy = float(trades["pnl"].mean())
        avg_bars = float(trades["bars_held"].mean()) if "bars_held" in trades.columns else 0
    else:
        n_trades = 0
        win_rate = 0
        avg_win = 0
        avg_loss = 0
        gross_profit = 0
        gross_loss = 0
        profit_factor = 0
        expectancy = 0
        avg_bars = 0

    calmar = (cagr / abs(max_dd)) if max_dd < 0 else 0.0

    return {
        "initial_capital": initial_capital,
        "final_equity": round(final, 2),
        "total_return_pct": round(total_return * 100, 2),
        "cagr_pct": round(cagr * 100, 2),
        "sharpe": round(sharpe, 3),
        "sharpe_active": round(sharpe_active, 3),
        "sortino": round(sortino, 3) if sortino != float("inf") else "inf",
        "calmar": round(calmar, 3),
        "max_drawdown_pct": round(max_dd * 100, 2),
        "max_dd_date": max_dd_date,
        "volatility_annual_pct": round(vol * 100, 2),
        "n_trades": n_trades,
        "win_rate_pct": round(win_rate * 100, 2),
        "avg_win": round(avg_win, 2),
        "avg_loss": round(avg_loss, 2),
        "profit_factor": round(profit_factor, 3) if profit_factor != float("inf") else "inf",
        "expectancy_per_trade": round(expectancy, 2),
        "avg_bars_held": round(avg_bars, 1),
    }