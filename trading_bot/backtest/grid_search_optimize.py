"""
Grid search walk-forward - Fix 2.

El workflow auto-mejora.yml invoca: python -m backtest.grid_search_optimize

Que hace:
  1. Carga datos del universo via funciones de run_v8_hedge_fund.
  2. Grid search sobre parametros de select_v8.
  3. Walk-forward: train 756 barras (3y), test 252 barras (1y),
     ventana rodante. Holdout final (ultimo 15%) NUNCA se toca.
  4. Guarda resultado en results/optimized_config.json
  5. Solo guarda si OOS Calmar mejora el umbral.
"""
from __future__ import annotations

import itertools
import json
import os
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RESULTS_DIR = os.path.join(ROOT, "results")
sys.path.insert(0, ROOT)

from backtest.run_v8_hedge_fund import (
    UNIVERSE_V8,
    build_momentum_table,
    select_v8,
    compute_metrics,
)

# ---------- parametros del grid ----------

GRID = {
    "top_n_base": [2, 3, 4],
    "max_crypto_weight": [0.20, 0.30, 0.40],
    "max_single_asset": [0.30, 0.40, 0.50],
    "bubble_penalty": [0.3, 0.5, 0.7],
}

# walk-forward
TRAIN_BARS = 756   # 3 anos de barras diarias
TEST_BARS = 252    # 1 ano
HOLDOUT_FRAC = 0.15  # ultimo 15% de la historia: intocable

# umbral: solo sugerir si OOS Calmar supera esto
MIN_OOS_CALMAR = 0.30


def load_all_data() -> pd.DataFrame:
    """Carga cierres diarios de todo el universo. Devuelve df de closes."""
    from backtest.run_v8_hedge_fund import load_tv_data
    closes = {}
    for sym, exch in UNIVERSE_V8:
        try:
            df = load_tv_data(sym, exch)
            if df is not None and len(df) > 100:
                closes[sym] = df["close"]
        except Exception as e:
            print(f"  {sym}: skip ({e})")
    all_df = pd.DataFrame(closes)
    all_df = all_df.dropna(how="all").sort_index()
    print(f"  Datos: {len(all_df)} dias x {all_df.shape[1]} simbolos")
    return all_df


def run_one_config(params: dict, closes: pd.DataFrame,
                   train_start, train_end) -> dict:
    """Corre una config en una ventana. Devuelve metricas dict."""
    window = closes.loc[train_start:train_end]
    if len(window) < 60:
        return {"cagr_pct": 0, "max_drawdown_pct": -100, "sharpe": 0, "calmar": 0}

    mom_df = build_momentum_table(UNIVERSE_V8)
    if mom_df is None or mom_df.empty:
        return {"cagr_pct": 0, "max_drawdown_pct": -100, "sharpe": 0, "calmar": 0}

    try:
        weights_df = select_v8(
            mom_df=mom_df,
            features=pd.DataFrame(),
            etf_scores={},
            top_n_base=params["top_n_base"],
            max_crypto_weight=params["max_crypto_weight"],
            max_single_asset=params["max_single_asset"],
            bubble_penalty=params["bubble_penalty"],
        )
        if weights_df is None or weights_df.empty:
            return {"cagr_pct": 0, "max_drawdown_pct": -100, "sharpe": 0, "calmar": 0}

        rets = window.pct_change().fillna(0)
        common = weights_df.index.intersection(rets.index)
        if len(common) < 30:
            return {"cagr_pct": 0, "max_drawdown_pct": -100, "sharpe": 0, "calmar": 0}

        w = weights_df.loc[common].shift(1).fillna(0)
        r = rets.loc[common]
        port_ret = (w * r).sum(axis=1).fillna(0)
        equity = (1 + port_ret).cumprod() * 500.0

        years = len(port_ret) / 252
        if years <= 0 or equity.iloc[-1] <= 0:
            return {"cagr_pct": 0, "max_drawdown_pct": -100, "sharpe": 0, "calmar": 0}
        cagr = ((equity.iloc[-1] / 500.0) ** (1 / years) - 1) * 100
        peak = equity.cummax()
        dd = (equity - peak) / peak
        max_dd = float(dd.min() * 100)
        sharpe = float(port_ret.mean() / port_ret.std() * np.sqrt(252)) if port_ret.std() > 0 else 0
        calmar = cagr / abs(max_dd) if max_dd != 0 else 0

        return {
            "cagr_pct": round(cagr, 2),
            "max_drawdown_pct": round(max_dd, 2),
            "sharpe": round(sharpe, 2),
            "calmar": round(calmar, 3),
            "final_equity": round(float(equity.iloc[-1]), 2),
        }
    except Exception as e:
        return {"cagr_pct": 0, "max_drawdown_pct": -100, "sharpe": 0, "calmar": 0}


def walk_forward(params: dict, closes: pd.DataFrame) -> dict:
    """
    Walk-forward: ventanas train/test rodantes.
    Devuelve promedio OOS de las ventanas test.
    Holdout final excluido del proceso.
    """
    total_bars = len(closes)
    holdout_start = int(total_bars * (1 - HOLDOUT_FRAC))
    wf_closes = closes.iloc[:holdout_start]

    oos_results = []
    start = 0
    while start + TRAIN_BARS + TEST_BARS <= len(wf_closes):
        train_end_idx = start + TRAIN_BARS
        test_start_idx = train_end_idx
        test_end_idx = min(test_start_idx + TEST_BARS, len(wf_closes))

        test_idx = wf_closes.index[test_start_idx:test_end_idx]

        m = run_one_config(params, wf_closes, test_idx[0], test_idx[-1])
        oos_results.append(m)
        start = test_end_idx

    if not oos_results:
        return {"cagr_pct": 0, "max_drawdown_pct": -100, "calmar": 0, "n_windows": 0}

    avg = {
        "cagr_pct": round(np.mean([r["cagr_pct"] for r in oos_results]), 2),
        "max_drawdown_pct": round(np.mean([r["max_drawdown_pct"] for r in oos_results]), 2),
        "sharpe": round(np.mean([r.get("sharpe", 0) for r in oos_results]), 2),
        "calmar": round(np.mean([r.get("calmar", 0) for r in oos_results]), 3),
        "n_windows": len(oos_results),
    }
    return avg


def main():
    print("=" * 60)
    print("Grid Search Walk-Forward")
    print(f"Train: {TRAIN_BARS} bars | Test: {TEST_BARS} bars | Holdout: {int(HOLDOUT_FRAC*100)}%")
    print("=" * 60)

    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("\n[1/3] Cargando datos...")
    closes = load_all_data()
    if len(closes) < TRAIN_BARS + TEST_BARS:
        print(f"ERROR: insuficiente historia ({len(closes)} < {TRAIN_BARS + TEST_BARS})")
        with open(os.path.join(RESULTS_DIR, "optimized_config.json"), "w") as f:
            json.dump({"status": "insufficient_data", "avg_oos_calmar": 0}, f, indent=2)
        return 0

    print("\n[2/3] Grid search...")
    keys = list(GRID.keys())
    combos = list(itertools.product(*[GRID[k] for k in keys]))
    print(f"  {len(combos)} combinaciones x walk-forward")

    results = []
    for i, combo in enumerate(combos):
        params = dict(zip(keys, combo))
        oos = walk_forward(params, closes)
        oos["params"] = params
        results.append(oos)
        if (i + 1) % 9 == 0 or i == len(combos) - 1:
            best_so_far = max(r["calmar"] for r in results)
            print(f"  [{i+1}/{len(combos)}] best Calmar so far: {best_so_far:.3f}")

    best = max(results, key=lambda r: r["calmar"])
    print(f"\n[3/3] Mejor config OOS:")
    print(f"  Calmar: {best['calmar']}")
    print(f"  CAGR: {best['cagr_pct']}%")
    print(f"  MaxDD: {best['max_drawdown_pct']}%")
    print(f"  Params: {best['params']}")

    if best["calmar"] >= MIN_OOS_CALMAR:
        output = {
            "status": "improved",
            "avg_oos_calmar": best["calmar"],
            "avg_oos_cagr_pct": best["cagr_pct"],
            "avg_oos_max_dd_pct": best["max_drawdown_pct"],
            "n_windows": best.get("n_windows", 0),
            "params": best["params"],
            "grid": GRID,
            "timestamp": pd.Timestamp.utcnow().isoformat(),
        }
        with open(os.path.join(RESULTS_DIR, "optimized_config.json"), "w") as f:
            json.dump(output, f, indent=2)
        print(f"\n  GUARDADO: results/optimized_config.json (Calmar {best['calmar']} >= {MIN_OOS_CALMAR})")
    else:
        output = {
            "status": "no_improvement",
            "avg_oos_calmar": best["calmar"],
            "threshold": MIN_OOS_CALMAR,
            "best_params": best["params"],
            "timestamp": pd.Timestamp.utcnow().isoformat(),
        }
        with open(os.path.join(RESULTS_DIR, "optimized_config.json"), "w") as f:
            json.dump(output, f, indent=2)
        print(f"\n  NO mejora (Calmar {best['calmar']} < {MIN_OOS_CALMAR}). Baseline se mantiene.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
