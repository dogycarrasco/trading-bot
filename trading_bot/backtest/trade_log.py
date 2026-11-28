"""
Trade-level logger para el backtest.

Produce:
  - results/trade_log.csv       (per-trade data con PnL, MAE/MFE, regime, scores)
  - results/equity_curve.csv    (equity mensual + drawdown)
  - results/backtest_config.json (params usados en el run)

CONDICIONES DEL REVIEWER:
  1. Solo instrumenta, NO cambia la logica. El equity/Calmar debe ser identico.
  2. Campos: exit_reason, holding_months, MAE, MFE, regime entrada/salida,
     momentum_at_entry, etf_score_at_entry.
  3. Equity curve aparte + config.

LIMITACION DOCUMENTADA:
  El backtest usa select_v8 (momentum + VIX + fundamentales).
  NO usa el MultiSignalEngine de 5 capas (regime/sentiment/onchain/trend/setup)
  del bot live. Por eso:
  - sentiment_vote, onchain_vote, setup_vote: NO disponibles en este log
  - exit_reason: solo 'rotation' (peso -> 0) o 'end_of_backtest'
  - MAE/MFE: computados con datos mensuales (no diarios)

  Para atribucion completa de 5 capas, habria que correr un backtest
  separado que use MultiSignalEngine. Esto es un primer paso.

NO LOOKAHEAD:
  Los scores (momentum, regime, etf_score) se registran al momento de
  entrada (mes de entry), no al de salida.

Uso:
    cd trading_bot
    python -m backtest.trade_log
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from datetime import datetime

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RESULTS_DIR = os.path.join(ROOT, "results")
CACHE_DIR = os.path.join(ROOT, "data", "tv_cache")
sys.path.insert(0, ROOT)

from backtest.run_v8_hedge_fund import (
    UNIVERSE_V8,
    load_tv_data,
    load_macro_features,
    build_momentum_table,
    get_all_etf_scores,
    select_v8,
    compute_metrics,
)


def load_monthly_closes():
    """Carga cierres diarios y resamplea a mensual."""
    monthly_data = {}
    for item in UNIVERSE_V8:
        sym, exch = item[0], item[1]
        for e in [exch, "AMEX", "NASDAQ", "CME_MINI", "NYMEX", "COMEX", "FX", "BINANCE"]:
            p = os.path.join(CACHE_DIR, f"{sym}_{e}.csv")
            if os.path.exists(p):
                try:
                    df = pd.read_csv(p, index_col=0, parse_dates=True)
                    monthly_close = df["close"].resample("ME").last()
                    monthly_data[sym] = monthly_close
                except Exception:
                    pass
                break
    closes = pd.DataFrame(monthly_data).sort_index()
    return closes


def compute_backtest_with_trades(
    weights_df, closes, mom_df, features, etf_scores,
    initial_capital=500.0, div_annual=0.010,
):
    """
    Replica exactamente backtest() PERO ademas trackea trades individuales.
    Devuelve: equity, monthly_returns, trades
    """
    rets = closes.pct_change()

    common_idx = weights_df.index.intersection(rets.index)
    weights_df = weights_df.loc[common_idx]
    rets = rets.loc[common_idx]

    common_cols = weights_df.columns.intersection(rets.columns)
    weights_df = weights_df[common_cols]
    rets = rets[common_cols]

    weights_shifted = weights_df.shift(1).fillna(0)
    port_returns = (weights_shifted * rets).sum(axis=1).fillna(0)
    equity = (1 + port_returns).cumprod() * initial_capital
    div_monthly = div_annual / 12
    equity = equity * (1 + div_monthly) ** np.arange(len(equity))

    # --- TRACKING DE TRADES (no afecta equity) ---
    trades = []
    holdings = {}

    for i, date in enumerate(weights_df.index):
        weights_row = weights_df.loc[date]
        rets_row = rets.loc[date] if date in rets.index else pd.Series(dtype=float)

        regime_val = None
        if features is not None and date in features.index:
            regime_val = features.loc[date].to_dict()

        for sym in weights_df.columns:
            w = float(weights_row.get(sym, 0.0))

            if sym not in holdings and w > 0:
                price = float(closes.loc[date, sym]) if sym in closes.columns and date in closes.index else None
                mom_val = float(mom_df.loc[date, sym]) if sym in mom_df.columns and date in mom_df.index else None
                etf_val = etf_scores.get(sym, None) if isinstance(etf_scores, dict) else None

                holdings[sym] = {
                    "symbol": sym,
                    "entry_date": date,
                    "entry_weight": w,
                    "entry_price": price,
                    "momentum_at_entry": mom_val,
                    "etf_score_at_entry": etf_val,
                    "regime_at_entry": regime_val,
                    "best_return": 0.0,
                    "worst_return": 0.0,
                    "months_held": 0,
                }

            if sym in holdings:
                holdings[sym]["months_held"] += 1
                if sym in rets_row.index:
                    r = float(rets_row[sym])
                    holdings[sym]["best_return"] = max(holdings[sym]["best_return"], r)
                    holdings[sym]["worst_return"] = min(holdings[sym]["worst_return"], r)

            if sym in holdings and w == 0:
                exit_price = float(closes.loc[date, sym]) if sym in closes.columns and date in closes.index else None
                entry_price = holdings[sym]["entry_price"]
                pnl_pct = (exit_price / entry_price - 1) * 100 if entry_price and exit_price else None

                trades.append({
                    **holdings[sym],
                    "exit_date": date,
                    "exit_price": exit_price,
                    "exit_weight": 0.0,
                    "pnl_pct": round(pnl_pct, 2) if pnl_pct is not None else None,
                    "mae_pct": round(holdings[sym]["worst_return"] * 100, 2),
                    "mfe_pct": round(holdings[sym]["best_return"] * 100, 2),
                    "holding_months": holdings[sym]["months_held"],
                    "exit_reason": "rotation",
                    "regime_at_exit": regime_val,
                })
                del holdings[sym]

    last_date = weights_df.index[-1]
    for sym, h in holdings.items():
        exit_price = float(closes.loc[last_date, sym]) if sym in closes.columns and last_date in closes.index else None
        entry_price = h["entry_price"]
        pnl_pct = (exit_price / entry_price - 1) * 100 if entry_price and exit_price else None

        trades.append({
            **h,
            "exit_date": last_date,
            "exit_price": exit_price,
            "exit_weight": 0.0,
            "pnl_pct": round(pnl_pct, 2) if pnl_pct is not None else None,
            "mae_pct": round(h["worst_return"] * 100, 2),
            "mfe_pct": round(h["best_return"] * 100, 2),
            "holding_months": h["months_held"],
            "exit_reason": "end_of_backtest",
            "regime_at_exit": None,
        })

    return equity, port_returns, trades


def export_results(equity, monthly_returns, trades, config, results_dir):
    os.makedirs(results_dir, exist_ok=True)

    trades_df = pd.DataFrame(trades)
    trade_path = os.path.join(results_dir, "trade_log.csv")
    trades_df.to_csv(trade_path, index=False)
    print(f"  trade_log.csv: {len(trades_df)} trades -> {trade_path}")

    eq_df = pd.DataFrame({
        "date": equity.index,
        "equity": equity.values,
    })
    eq_df["peak"] = equity.cummax().values
    eq_df["drawdown_pct"] = ((equity / equity.cummax() - 1) * 100).values
    eq_path = os.path.join(results_dir, "equity_curve.csv")
    eq_df.to_csv(eq_path, index=False)
    print(f"  equity_curve.csv: {len(eq_df)} meses -> {eq_path}")

    cfg_path = os.path.join(results_dir, "backtest_config.json")
    with open(cfg_path, "w") as f:
        json.dump(config, f, indent=2, default=str)
    print(f"  backtest_config.json -> {cfg_path}")

    metrics = compute_metrics(equity, monthly_returns)
    print(f"\n  Verificacion (debe coincidir con backtest sin logging):")
    print(f"    CAGR: {metrics['cagr_pct']}%")
    print(f"    MaxDD: {metrics['max_drawdown_pct']}%")
    print(f"    Sharpe: {metrics['sharpe']}")
    print(f"    Final equity: ${metrics['final_equity']}")

    if len(trades_df) > 0:
        winners = trades_df[trades_df["pnl_pct"] > 0]
        losers = trades_df[trades_df["pnl_pct"] < 0]
        print(f"\n  Trade summary:")
        print(f"    Total trades: {len(trades_df)}")
        print(f"    Winners: {len(winners)} ({len(winners)/len(trades_df)*100:.1f}%)")
        print(f"    Losers: {len(losers)} ({len(losers)/len(trades_df)*100:.1f}%)")
        if len(winners) > 0:
            print(f"    Avg win: +{winners['pnl_pct'].mean():.2f}%")
        if len(losers) > 0:
            print(f"    Avg loss: {losers['pnl_pct'].mean():.2f}%")
        if len(losers) > 0 and len(winners) > 0:
            rr = winners["pnl_pct"].mean() / abs(losers["pnl_pct"].mean())
            print(f"    Win/Loss ratio: {rr:.2f}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--top_n_base", type=int, default=2)
    p.add_argument("--max_crypto_weight", type=float, default=0.20)
    p.add_argument("--max_single_asset", type=float, default=0.30)
    p.add_argument("--bubble_penalty", type=float, default=0.3)
    p.add_argument("--initial_capital", type=float, default=500.0)
    args = p.parse_args()

    print("=" * 60)
    print("Trade Log Generator")
    print("=" * 60)

    print("\n[1/4] Cargando datos...")
    closes = load_monthly_closes()
    print(f"  closes: {closes.shape}")

    print("\n[2/4] Cargando features + momentum + ETF scores...")
    features = load_macro_features()
    etf_scores = get_all_etf_scores()
    mom_df = build_momentum_table(UNIVERSE_V8)
    print(f"  momentum: {mom_df.shape}, features: {features.shape if features is not None else 'None'}")

    print("\n[3/4] Running select_v8 + backtest with trade logging...")
    result = select_v8(
        mom_df=mom_df,
        features=features,
        etf_scores=etf_scores,
        top_n_base=args.top_n_base,
        max_crypto_weight=args.max_crypto_weight,
        max_single_asset=args.max_single_asset,
        bubble_penalty=args.bubble_penalty,
    )
    weights_df = result[0] if isinstance(result, tuple) else result

    if weights_df is None or (hasattr(weights_df, "empty") and weights_df.empty):
        print("  ERROR: select_v8 returned empty weights")
        return 1

    equity, monthly_returns, trades = compute_backtest_with_trades(
        weights_df=weights_df,
        closes=closes,
        mom_df=mom_df,
        features=features,
        etf_scores=etf_scores,
        initial_capital=args.initial_capital,
    )

    config = {
        "params": {
            "top_n_base": args.top_n_base,
            "max_crypto_weight": args.max_crypto_weight,
            "max_single_asset": args.max_single_asset,
            "bubble_penalty": args.bubble_penalty,
        },
        "initial_capital": args.initial_capital,
        "universe_size": len(UNIVERSE_V8),
        "n_months": len(equity),
        "timestamp": datetime.utcnow().isoformat(),
        "note": "Backtest usa select_v8 (momentum+VIX+fundamentals). No incluye 5-layer engine.",
    }

    print("\n[4/4] Exportando resultados...")
    export_results(equity, monthly_returns, trades, config, RESULTS_DIR)

    print("\n" + "=" * 60)
    print("DONE. Archivos en results/:")
    print("  - trade_log.csv")
    print("  - equity_curve.csv")
    print("  - backtest_config.json")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
