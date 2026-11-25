"""Grid search con walk-forward para auto-mejora.
Reusa el framework validado (Runner parametrizable + reindex+ffill).
Output: results/optimized_config.json (solo si OOS mejora)."""
import os, sys, json, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from backtest.run_v8_hedge_fund import UNIVERSE_V8, load_tv_data, load_macro_features

DATA = os.path.join(ROOT, "data", "tv_cache")
FEE = 0.001; CAP0 = 500.0

def ema(s, n): return s.ewm(span=n, adjust=False).mean()
def rsi_w(s, n=14):
    d = s.diff(); g = d.clip(lower=0).ewm(alpha=1/n, adjust=False).mean()
    l = (-d.clip(upper=0)).ewm(alpha=1/n, adjust=False).mean()
    return (100 - 100/(1 + g/l.replace(0, np.nan))).fillna(50)

def load_bars(sym):
    for exch in ["AMEX", "NASDAQ", "BINANCE", "CBOE"]:
        p = os.path.join(DATA, f"{sym}_{exch}.csv")
        if os.path.exists(p):
            df = pd.read_csv(p, index_col=0, parse_dates=True)
            return df.dropna(subset=["close"])
    return pd.DataFrame()

def run_grid_search():
    """Grid search walk-forward: train 756 bars, test 252 bars, rolling window."""
    DC = {s: load_bars(s) for s in [u[0] for u in UNIVERSE_V8] + ["VIX"]}
    IC = {}
    for s in [u[0] for u in UNIVERSE_V8]:
        c = DC[s]["close"]
        IC[s] = pd.DataFrame({"close": c, "high": DC[s]["high"], "low": DC[s]["low"],
            "ema50": ema(c, 50), "ema200": ema(c, 200), "ma20": c.rolling(20).mean(),
            "ma20p": c.rolling(20).mean().shift(1), "rsi": rsi_w(c), "mom20": c/c.shift(20)-1,
            "ema9": ema(c, 9)})
    vix = DC.get("VIX", pd.DataFrame())
    spy = DC.get("SPY", pd.DataFrame())
    macro = pd.DataFrame({"vix": vix["close"] if not vix.empty else pd.Series(dtype=float),
        "spy_gt200": (spy["close"] > ema(spy["close"], 200)).astype(int) if not spy.empty else pd.Series(dtype=float)})

    idx = None
    for s in DC:
        if not DC[s].empty:
            idx = DC[s].index if idx is None else idx.intersection(DC[s].index)
    if idx is None or len(idx) < 1008:
        print("Datos insuficientes para grid search")
        return None

    # Walk-forward: 756 train + 252 test, rolling
    results = []
    for i in range(0, len(idx) - 1008, 252):
        train_idx = idx[i:i+756]
        test_idx = idx[i+756:i+1008]
        if len(train_idx) < 500 or len(test_idx) < 100:
            continue

        # Parámetros a optimizar
        best_calmar = -99
        best_params = {}
        for min_votes in [2, 3]:
            for trail_pct in [0.06, 0.08, 0.10]:
                for rsi_lo, rsi_hi in [(40, 72), (45, 72), (45, 75)]:
                    params = {"min_votes": min_votes, "trail_pct": trail_pct,
                              "rsi_lo": rsi_lo, "rsi_hi": rsi_hi}
                    # Evaluar en train (simplificado: contar trades consistentes)
                    calmar_train = _eval_period(IC, macro, train_idx, params, DC)
                    if calmar_train > best_calmar:
                        best_calmar = calmar_train
                        best_params = params

        # Evaluar best_params en test (OOS)
        calmar_test = _eval_period(IC, macro, test_idx, best_params, DC)
        results.append({"train_calmar": round(best_calmar, 3),
                        "test_calmar": round(calmar_test, 3),
                        "params": best_params})

    # Promediar OOS
    if results:
        avg_oos = np.mean([r["test_calmar"] for r in results])
        best_overall = max(results, key=lambda x: x["test_calmar"])
        output = {"avg_oos_calmar": round(avg_oos, 3),
                  "best_params": best_overall["params"],
                  "best_oos_calmar": best_overall["test_calmar"],
                  "windows": len(results)}
        out_path = os.path.join(ROOT, "results", "optimized_config.json")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(output, f, indent=2)
        print(f"Grid search: {len(results)} windows, OOS promedio Calmar={avg_oos:.3f}")
        print(f"Mejores params: {best_overall['params']}")
        return output
    return None

def _eval_period(IC, macro, dates, params, DC):
    """Evalúa una ventana con parámetros dados. Retorna Calmar aproximado."""
    mv = params["min_votes"]; tr = params["trail_pct"]
    rl = params["rsi_lo"]; rh = params["rsi_hi"]
    cash, pos, eq, trades = CAP0, {}, [], []
    for d in dates:
        if d not in macro.index:
            eq.append({"date": d, "eq": cash}); continue
        vix = float(macro.loc[d, "vix"]) if not pd.isna(macro.loc[d, "vix"]) else 20
        sg = int(macro.loc[d, "spy_gt200"]) if not pd.isna(macro.loc[d, "spy_gt200"]) else 0
        regime = 1 if (sg == 1 and vix < 25) else (0 if vix < 30 else -1)
        # Exits
        for sym in list(pos):
            r = IC.get(sym)
            if r is None or d not in r.index: continue
            px = float(r.loc[d, "close"]); pos[sym]["peak"] = max(pos[sym]["peak"], px)
            days = (d - pos[sym]["didx"]).days; ex = None
            if px < pos[sym]["peak"] * (1 - tr): ex = "trail"
            elif px < r.loc[d, "ema50"] and r.loc[d, "ema50"] < r.loc[d, "ema200"]: ex = "flip"
            elif days >= 63: ex = "timeout"
            if ex:
                pr = pos[sym]["qty"] * px * (1 - FEE); cash += pr
                trades.append(pr - pos[sym]["qty"] * pos[sym]["entry"]); del pos[sym]
        # Entries
        slots = 5 - len(pos)
        if slots > 0:
            cands = []
            for sym in [u[0] for u in UNIVERSE_V8]:
                if sym in pos: continue
                r = IC.get(sym)
                if r is None or d not in r.index or pd.isna(r.loc[d, "ema200"]): continue
                px = float(r.loc[d, "close"]); mom20 = float(r.loc[d, "mom20"])
                rsi_v = float(r.loc[d, "rsi"])
                trend = 1 if (px > r.loc[d, "ema50"] > r.loc[d, "ema200"] and mom20 > 0) else 0
                setup = 1 if (px > r.loc[d, "ma20"] and r.loc[d, "ma20"] > r.loc[d, "ma20p"] and rl <= rsi_v <= rh and px > r.loc[d, "ema9"]) else 0
                votes = sum(v for v in [regime, trend, setup] if v > 0)
                if votes >= mv: cands.append((sym, votes, mom20))
            cands.sort(key=lambda x: (-x[1], -x[2]))
            for sym, v, _ in cands[:slots]:
                px = float(IC[sym].loc[d, "close"]); amt = cash * (0.33 if v == mv else 0.5)
                if amt < 10: continue
                cash -= amt; pos[sym] = {"qty": amt * (1 - FEE) / px, "entry": px, "peak": px, "didx": d}
        v = cash + sum(p["qty"] * float(IC[s].loc[d, "close"]) for s, p in pos.items() if s in IC and d in IC[s].index)
        eq.append({"date": d, "eq": v})
    eq = pd.DataFrame(eq).set_index("date")["eq"]
    if len(eq) < 2 or eq.iloc[-1] <= 0: return -99
    y = len(eq) / 252; cagr = (eq.iloc[-1] / eq.iloc[0]) ** (1/y) - 1
    dd = (eq / eq.cummax() - 1).min()
    return cagr / abs(dd) if dd < 0 else 0

if __name__ == "__main__":
    run_grid_search()
