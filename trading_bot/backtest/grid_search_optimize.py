"""
Grid search walk-forward - Fix 2.

python -m backtest.grid_search_optimize
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
    load_macro_features,
    get_all_etf_scores,
)

GRID = {
    "top_n_base": [2, 3, 4],
    "max_crypto_weight": [0.20, 0.30, 0.40],
    "max_single_asset": [0.30, 0.40, 0.50],
    "bubble_penalty": [0.3, 0.5, 0.7],
}

# walk-forward MENSUAL (no diario)
TRAIN_BARS = 24   # 2 anos en meses
TEST_BARS = 6     # 6 meses en meses
HOLDOUT_FRAC = 0.20
MIN_OOS_CALMAR = 0.30



def count_walk_forward_windows(n_bars, train_bars=TRAIN_BARS, test_bars=TEST_BARS, holdout_frac=HOLDOUT_FRAC):
    """Numero de ventanas OOS del walk-forward (rolling por test_bars)."""
    holdout_start = int(n_bars * (1 - holdout_frac))
    n = 0
    start = 0
    while start + train_bars + test_bars <= holdout_start:
        n += 1
        start += test_bars
    return n

def load_all_data():
    """Carga cierres diarios del universo y convierte a mensual."""
    from backtest.run_v8_hedge_fund import load_tv_data
    closes = {}
    for item in UNIVERSE_V8:
        sym, exch = item[0], item[1]
        try:
            df = load_tv_data(sym, exch)
            if df is not None and len(df) > 100:
                closes[sym] = df["close"]
        except Exception as e:
            print(f"  {sym}: skip ({e})")
    daily_df = pd.DataFrame(closes)
    daily_df = daily_df.dropna(how="all").sort_index()
    # convertir a mensual (ultimo close de cada mes)
    monthly_df = daily_df.resample("ME").last()
    print(f"  Datos: {len(daily_df)} dias -> {len(monthly_df)} meses x {monthly_df.shape[1]} simbolos")
    return monthly_df


def compute_window_metrics(weights_df, rets, test_idx):
    """Evalua una ventana test mensual."""
    if weights_df is None:
        return None
    if hasattr(weights_df, 'empty') and weights_df.empty:
        return None

    w = weights_df.reindex(test_idx).shift(1).fillna(0)
    r = rets.reindex(test_idx).fillna(0)

    common_symbols = w.columns.intersection(r.columns)
    if len(common_symbols) == 0:
        return None

    w = w[common_symbols]
    r = r[common_symbols]
    port_ret = (w * r).sum(axis=1).fillna(0)

    if len(port_ret) < 6 or port_ret.std() == 0:
        return None

    equity = (1 + port_ret).cumprod() * 500.0
    years = len(port_ret) / 12
    if years <= 0 or equity.iloc[-1] <= 0:
        return None

    cagr = ((equity.iloc[-1] / 500.0) ** (1 / years) - 1) * 100
    peak = equity.cummax()
    dd = (equity - peak) / peak
    max_dd = float(dd.min() * 100)
    sharpe = float(port_ret.mean() / port_ret.std() * np.sqrt(12)) if port_ret.std() > 0 else 0
    calmar = cagr / abs(max_dd) if max_dd != 0 else 0

    return {
        "cagr_pct": round(cagr, 2),
        "max_drawdown_pct": round(max_dd, 2),
        "sharpe": round(sharpe, 2),
        "calmar": round(calmar, 3),
    }


def walk_forward_one_combo(params, mom_df, features, etf_scores, closes, rets):
    """Para una config: correr select_v8 una vez, evaluar en ventanas test."""
    try:
        result = select_v8(
            mom_df=mom_df,
            features=features,
            etf_scores=etf_scores,
            top_n_base=params["top_n_base"],
            max_crypto_weight=params["max_crypto_weight"],
            max_single_asset=params["max_single_asset"],
            bubble_penalty=params["bubble_penalty"],
        )
    except Exception as e:
        return {"calmar": 0, "cagr_pct": 0, "max_drawdown_pct": -100, "n_windows": 0,
                "error": f"select_v8 raised: {e}"}

    # select_v8 puede devolver tuple (weights, selections) o solo weights
    if isinstance(result, tuple):
        weights_df = result[0]
    else:
        weights_df = result

    if weights_df is None:
        return {"calmar": 0, "cagr_pct": 0, "max_drawdown_pct": -100, "n_windows": 0,
                "error": "select_v8 returned None"}
    if hasattr(weights_df, 'empty') and weights_df.empty:
        return {"calmar": 0, "cagr_pct": 0, "max_drawdown_pct": -100, "n_windows": 0,
                "error": "select_v8 returned empty"}

    total_bars = len(closes)
    holdout_start = int(total_bars * (1 - HOLDOUT_FRAC))

    oos_results = []
    start = 0
    while start + TRAIN_BARS + TEST_BARS <= holdout_start:
        test_start = start + TRAIN_BARS
        test_end = min(test_start + TEST_BARS, holdout_start)
        test_idx = closes.index[test_start:test_end]

        m = compute_window_metrics(weights_df, rets, test_idx)
        if m is not None:
            oos_results.append(m)
        start += TEST_BARS

    if not oos_results:
        return {"calmar": 0, "cagr_pct": 0, "max_drawdown_pct": -100, "n_windows": 0,
                "error": "no valid windows (weights index mismatch?)"}

    if len(oos_results) < 3:
        return {"calmar": 0, "cagr_pct": 0, "max_drawdown_pct": 0, "n_windows": len(oos_results),
                "status": "insufficient_windows"}

    return {
        "cagr_pct": round(np.mean([r["cagr_pct"] for r in oos_results]), 2),
        "max_drawdown_pct": round(np.mean([r["max_drawdown_pct"] for r in oos_results]), 2),
        "sharpe": round(np.mean([r.get("sharpe", 0) for r in oos_results]), 2),
        "calmar": round(np.mean([r.get("calmar", 0) for r in oos_results]), 3),
        "n_windows": len(oos_results),
    }


def main():
    print("=" * 60)
    print("Grid Search Walk-Forward (mensual)")
    print(f"Train: {TRAIN_BARS} meses | Test: {TEST_BARS} meses | Holdout: {int(HOLDOUT_FRAC*100)}%")
    print("=" * 60)

    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("\n[1/4] Cargando datos...")
    closes = load_all_data()
    if len(closes) < TRAIN_BARS + TEST_BARS:
        print(f"ERROR: insuficiente historia ({len(closes)} meses < {TRAIN_BARS + TEST_BARS})")
        with open(os.path.join(RESULTS_DIR, "optimized_config.json"), "w") as f:
            json.dump({"status": "insufficient_data", "avg_oos_calmar": 0}, f, indent=2)
        return 0

    print("\n[2/4] Cargando features + momentum...")
    features = load_macro_features()
    etf_scores = get_all_etf_scores()
    mom_df = build_momentum_table(UNIVERSE_V8)
    rets = closes.pct_change().fillna(0)

    if mom_df is None or (hasattr(mom_df, 'empty') and mom_df.empty):
        print("ERROR: momentum table vacia")
        with open(os.path.join(RESULTS_DIR, "optimized_config.json"), "w") as f:
            json.dump({"status": "no_momentum", "avg_oos_calmar": 0}, f, indent=2)
        return 0

    print(f"  momentum: {mom_df.shape}, features: {features.shape if features is not None else 'None'}")
    print(f"  etf_scores: {len(etf_scores) if etf_scores else 0} simbolos")
    print(f"  closes: {closes.shape}, rets: {rets.shape}")

    print("\n[3/4] Grid search...")
    keys = list(GRID.keys())
    combos = list(itertools.product(*[GRID[k] for k in keys]))
    print(f"  {len(combos)} combinaciones")

    results = []
    first_error = None
    for i, combo in enumerate(combos):
        params = dict(zip(keys, combo))
        oos = walk_forward_one_combo(params, mom_df, features, etf_scores, closes, rets)
        oos["params"] = params
        results.append(oos)

        if oos.get("error") and first_error is None:
            first_error = oos["error"]
            print(f"  [WARN] combo {i+1} ({params}): {oos['error']}")

        if (i + 1) % 9 == 0 or i == len(combos) - 1:
            valid = [r for r in results if r["calmar"] > 0]
            best_so_far = max((r["calmar"] for r in valid), default=0)
            print(f"  [{i+1}/{len(combos)}] best Calmar: {best_so_far:.3f} ({len(valid)} valid)")

    valid_results = [r for r in results if r["calmar"] > 0]
    if not valid_results:
        print("\n  TODAS las combinaciones dieron error o Calmar 0.")
        if first_error:
            print(f"  Primer error: {first_error}")
        output = {
            "status": "all_failed",
            "avg_oos_calmar": 0,
            "first_error": first_error,
            "timestamp": pd.Timestamp.utcnow().isoformat(),
        }
        with open(os.path.join(RESULTS_DIR, "optimized_config.json"), "w") as f:
            json.dump(output, f, indent=2)
        return 0

    best = max(valid_results, key=lambda r: r["calmar"])
    print(f"\n[4/4] Mejor config OOS:")
    print(f"  Calmar: {best['calmar']}")
    print(f"  CAGR: {best['cagr_pct']}%")
    print(f"  MaxDD: {best['max_drawdown_pct']}%")
    print(f"  Windows: {best.get('n_windows', 0)}")
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
        print(f"\n  GUARDADO: Calmar {best['calmar']} >= {MIN_OOS_CALMAR}")
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
        print(f"\n  NO mejora (Calmar {best['calmar']} < {MIN_OOS_CALMAR}).")

    return 0


if __name__ == "__main__":
    sys.exit(main())
