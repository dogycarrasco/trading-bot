import numpy as np
import pandas as pd

def vectorbt_backtest(prices, signals, initial_capital=500.0):
    try:
        import vectorbt as vbt
    except ImportError:
        return None
    try:
        entries = signals == 1
        exits = signals == 0
        pf = vbt.Portfolio.from_signals(prices, entries, exits, init_cash=initial_capital, fees=0.001, slippage=0.0005, freq="D")
        equity = pf.value()
        returns = pf.returns()
        metrics = {"total_return":float(pf.total_return()*100),"sharpe":float(pf.sharpe_ratio()),"max_drawdown":float(pf.max_drawdown()*100),"win_rate":float(pf.trades.win_rate()*100),"n_trades":int(pf.trades.count())}
        return {"equity":equity,"returns":returns,"metrics":metrics}
    except Exception as e:
        print(f"VectorBT error: {e}")
        return None
