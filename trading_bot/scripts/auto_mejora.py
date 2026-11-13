"""
Auto-mejora continua - OODA loop automatico semanal.
"""
import os
import sys
import json
import warnings
import time
import random
import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
TRADING_ROOT = os.path.dirname(ROOT)
sys.path.insert(0, TRADING_ROOT)

STATE_PATH = os.path.join(TRADING_ROOT, "results", "live_state_v8.json")
REPORT_PATH = os.path.join(TRADING_ROOT, "results", "auto_mejora_report.json")
PARAMS_PATH = os.path.join(TRADING_ROOT, "results", "best_params.json")
HISTORY_PATH = os.path.join(TRADING_ROOT, "results", "auto_mejora_history.json")


def load_state():
    if not os.path.exists(STATE_PATH):
        return None
    try:
        with open(STATE_PATH) as f:
            return json.load(f)
    except Exception:
        return None


def load_history():
    if not os.path.exists(HISTORY_PATH):
        return []
    try:
        with open(HISTORY_PATH) as f:
            return json.load(f)
    except Exception:
        return []


def save_history(history):
    os.makedirs(os.path.dirname(HISTORY_PATH), exist_ok=True)
    with open(HISTORY_PATH, "w") as f:
        json.dump(history, f, indent=2, default=str)


def measure_live_performance(state):
    if not state:
        return None
    history = state.get("history", [])
    weekly_stats = state.get("weekly_stats", [])
    cutoff = datetime.now(timezone.utc) - timedelta(days=14)
    recent_trades = []
    for t in history:
        try:
            t_date = datetime.fromisoformat(t["date"].replace("Z", "+00:00"))
            if t_date >= cutoff:
                recent_trades.append(t)
        except Exception:
            continue
    closed_trades = [t for t in recent_trades if t.get("action") == "SELL"]
    wins = [t for t in closed_trades if (t.get("pnl") or 0) > 0]
    win_rate = len(wins) / len(closed_trades) * 100 if closed_trades else 0
    total_pnl = sum(t.get("pnl", 0) for t in closed_trades)
    avg_pnl = total_pnl / len(closed_trades) if closed_trades else 0
    recent_weekly = weekly_stats[-4:] if weekly_stats else []
    positive_weeks = sum(1 for w in recent_weekly if (w.get("current_pnl") or 0) > 0)
    weekly_wr = positive_weeks / len(recent_weekly) * 100 if recent_weekly else 0
    cash = state.get("cash", 0)
    positions = state.get("positions", {})
    total_value = cash + sum(p.get("value", 0) for p in positions.values())
    initial = 500
    total_return = (total_value / initial - 1) * 100 if initial > 0 else 0
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "total_value": total_value,
        "total_return_pct": total_return,
        "trades_last_2_weeks": len(closed_trades),
        "win_rate_2w": win_rate,
        "total_pnl_2w": total_pnl,
        "avg_pnl_per_trade": avg_pnl,
        "weekly_wr_recent": weekly_wr,
        "n_weekly_samples": len(recent_weekly),
        "positions_count": len(positions),
        "n_total_history": len(history),
    }


def generate_param_mutations(base_params, n_variants=5):
    variants = []
    for i in range(n_variants):
        v = dict(base_params)
        n_mutations = random.choice([1, 2])
        mutable_keys = ["min_votes", "pt_factor", "sl_factor", "max_hold_days",
                        "circuit_breaker_dd", "trailing_trigger"]
        chosen = random.sample(mutable_keys, n_mutations)
        for key in chosen:
            if key in v:
                if isinstance(v[key], int):
                    delta = random.choice([-1, 1])
                    v[key] = max(1, v[key] + delta)
                elif isinstance(v[key], float):
                    delta = v[key] * random.uniform(-0.2, 0.2)
                    v[key] = max(0.01, v[key] + delta)
        v["variant_name"] = f"variant_{i+1}"
        v["mutations"] = chosen
        variants.append(v)
    return variants


def backtest_variant(params, days=252):
    try:
        from backtest.run_v8_hedge_fund import (
            UNIVERSE_V8, load_tv_data, load_macro_features,
        )
        from features.signal_engine import MultiSignalEngine
        from features.triple_barrier import (
            atr_at_date, compute_barriers, evaluate_trade_outcome,
        )

        engine = MultiSignalEngine(min_votes_to_buy=int(params.get("min_votes", 3)))
        pt_factor = float(params.get("pt_factor", 5.0))
        sl_factor = float(params.get("sl_factor", 5.0))
        max_hold = int(params.get("max_hold_days", 63))

        asset_dfs = {}
        for sym, _, _ in UNIVERSE_V8:
            df = load_tv_data(sym, "")
            if not df.empty:
                asset_dfs[sym] = df
        if not asset_dfs:
            return None

        all_dates = sorted(set().union(*[set(df.index) for df in asset_dfs.values()]))
        if len(all_dates) < days + 50:
            days = len(all_dates) - 50
        recent_dates = all_dates[-days:]

        cash = 500.0
        positions = {}
        trade_history = []
        equity_curve = []

        for date in recent_dates:
            ts = pd.Timestamp(date)
            total_value = cash
            for sym, pos in list(positions.items()):
                df = asset_dfs.get(sym)
                if df is None or date not in df.index:
                    continue
                price = float(df.loc[date, "close"])
                pos["current_price"] = price
                pos["value"] = pos["qty"] * price
                total_value += pos["value"]

            for sym, pos in list(positions.items()):
                df = asset_dfs.get(sym)
                if df is None or date not in df.index:
                    continue
                barriers = pos.get("barriers")
                if not barriers:
                    continue
                price_today = float(df.loc[date, "close"])
                entry = pos["entry_price"]
                outcome = evaluate_trade_outcome(
                    df.loc[:date], pos["entry_date"], entry, barriers,
                    max_hold_days=max_hold,
                )
                if outcome is None:
                    continue
                should_exit = False
                exit_price = price_today
                if outcome["outcome"] == +1:
                    should_exit = True
                    exit_price = barriers["tp_price"]
                elif outcome["outcome"] == -1:
                    should_exit = True
                    exit_price = barriers["sl_price"]
                elif outcome["hold_days"] >= max_hold:
                    should_exit = True
                    exit_price = price_today
                peak_pos = pos.get("peak_price", entry)
                if price_today > peak_pos:
                    pos["peak_price"] = price_today
                    peak_pos = price_today
                trigger = float(params.get("trailing_trigger", 0.10))
                drop = float(params.get("trailing_drop", -0.05))
                gain = (peak_pos / entry - 1) if entry > 0 else 0
                drop_from_peak = (price_today / peak_pos - 1) if peak_pos > 0 else 0
                if gain >= trigger and drop_from_peak <= drop:
                    should_exit = True
                    exit_price = price_today
                if should_exit:
                    pnl = (exit_price - entry) * pos["qty"]
                    cash += pos["qty"] * exit_price
                    trade_history.append({"pnl": pnl})
                    del positions[sym]

            if cash > 10 and len(positions) < 5:
                try:
                    macro_votes = engine.evaluate_macro(ts)
                except Exception:
                    continue
                for sym, _, _ in UNIVERSE_V8:
                    if sym in positions:
                        continue
                    df = asset_dfs.get(sym)
                    if df is None or date not in df.index:
                        continue
                    try:
                        result = engine.evaluate_asset(ts, df, sym, macro_votes)
                    except Exception:
                        continue
                    if result["decision"] != "BUY":
                        continue
                    sub = df.loc[:date]
                    price = float(sub["close"].iloc[-1])
                    if price <= 0:
                        continue
                    atr_val = atr_at_date(df, ts)
                    barriers = compute_barriers(price, atr_val,
                                                 pt_factor=pt_factor,
                                                 sl_factor=sl_factor,
                                                 max_hold_days=max_hold)
                    amount = min(cash * 0.30, cash)
                    if amount < 10:
                        continue
                    qty = amount / price
                    positions[sym] = {
                        "qty": qty, "entry_price": price,
                        "entry_date": date, "barriers": barriers,
                        "peak_price": price,
                    }
                    cash -= amount
            equity_curve.append({"date": date, "value": total_value})

        if len(equity_curve) < 30:
            return None

        eq = pd.DataFrame(equity_curve).set_index("date")["value"]
        returns = eq.pct_change().dropna()
        final_value = cash + sum(p.get("value", 0) for p in positions.values())
        n_years = (eq.index[-1] - eq.index[0]).days / 365.25
        ratio = final_value / 500
        cagr = (ratio ** (1/n_years) - 1) * 100 if ratio > 0 and n_years > 0 else 0
        peak_eq = eq.cummax()
        drawdown = ((eq / peak_eq - 1) * 100).min()
        sharpe = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
        wins = sum(1 for t in trade_history if t["pnl"] > 0)
        wr = wins / len(trade_history) * 100 if trade_history else 0
        return {"final_value": final_value, "cagr": cagr, "max_dd": drawdown,
                "sharpe": sharpe, "win_rate": wr, "n_trades": len(trade_history)}
    except Exception as e:
        return {"error": str(e)}


def detect_alpha_decay(live_perf, baseline_wr=44.9):
    if not live_perf or live_perf["trades_last_2_weeks"] < 5:
        return {"decay": False, "reason": "insufficient_data", "severity": "none"}
    live_wr = live_perf["weekly_wr_recent"]
    if live_wr < baseline_wr - 10:
        return {"decay": True, "reason": f"weekly_wr {live_wr:.1f}% << baseline {baseline_wr}%",
                "severity": "high"}
    elif live_wr < baseline_wr - 5:
        return {"decay": True, "reason": f"weekly_wr {live_wr:.1f}% < baseline {baseline_wr}%",
                "severity": "medium"}
    return {"decay": False, "reason": f"weekly_wr {live_wr:.1f}% ok",
            "severity": "none"}


def save_report(report):
    os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
    with open(REPORT_PATH, "w") as f:
        json.dump(report, f, indent=2, default=str)
    history = load_history()
    history.append(report)
    if len(history) > 52:
        history = history[-52:]
    save_history(history)


def main():
    print("=" * 70)
    print("AUTO-MEJORA SEMANAL - OODA Loop")
    print("=" * 70)

    BASELINE_PARAMS = {
        "min_votes": 3, "pt_factor": 5.0, "sl_factor": 5.0,
        "max_hold_days": 63, "circuit_breaker_dd": -0.15,
        "trailing_trigger": 0.10, "trailing_drop": -0.05,
    }

    print("\n[1] OBSERVE: midiando performance live...")
    state = load_state()
    live_perf = measure_live_performance(state)
    if live_perf:
        print(f"  Total value: ${live_perf['total_value']:.2f}")
        print(f"  Return: {live_perf['total_return_pct']:+.2f}%")
        print(f"  Trades 2w: {live_perf['trades_last_2_weeks']}")
        print(f"  Win rate 2w: {live_perf['win_rate_2w']:.1f}%")
        print(f"  Win rate semanal: {live_perf['weekly_wr_recent']:.1f}%")

    print("\n[2] ORIENT: detectando alpha decay...")
    decay = detect_alpha_decay(live_perf)
    print(f"  Decay: {decay['decay']} ({decay['severity']}) - {decay['reason']}")

    print("\n[3] DECIDE: generando variantes mutadas...")
    random.seed(42)
    variants = generate_param_mutations(BASELINE_PARAMS, n_variants=5)
    print(f"  {len(variants)} variantes generadas")

    print("\n[4] ACT: backtest de cada variante...")
    results = []
    for v in variants:
        print(f"  {v['variant_name']} (mutations: {v['mutations']})...")
        t0 = time.time()
        metrics = backtest_variant(v, days=252)
        elapsed = time.time() - t0
        if metrics and "error" not in metrics:
            metrics["variant"] = v["variant_name"]
            metrics["mutations"] = v["mutations"]
            metrics["elapsed_s"] = elapsed
            results.append(metrics)
            print(f"    CAGR {metrics['cagr']:+.2f}%, win {metrics['win_rate']:.1f}%, "
                  f"trades {metrics['n_trades']}, {elapsed:.0f}s")

    print("\n[5] Comparando con baseline...")
    baseline_metrics = backtest_variant(BASELINE_PARAMS, days=252)
    if baseline_metrics:
        print(f"  Baseline: CAGR {baseline_metrics.get('cagr', 0):+.2f}%, "
              f"win {baseline_metrics.get('win_rate', 0):.1f}%")

    best_variant = None
    if results and baseline_metrics:
        baseline_score = (baseline_metrics.get("cagr", 0) * 0.5 +
                          baseline_metrics.get("win_rate", 0) * 0.3 +
                          baseline_metrics.get("sharpe", 0) * 20 * 0.2)
        for r in results:
            score = r.get("cagr", 0) * 0.5 + r.get("win_rate", 0) * 0.3 + r.get("sharpe", 0) * 20 * 0.2
            r["score"] = score
        results.sort(key=lambda x: -x["score"])
        best_variant = results[0]
        print(f"\n  Mejor: {best_variant['variant']} (score {best_variant['score']:+.2f})")
        print(f"    CAGR {best_variant['cagr']:+.2f}% vs baseline {baseline_metrics.get('cagr', 0):+.2f}%")

        if best_variant["score"] > baseline_score * 1.05:
            print(f"\n  >>> MEJORA ENCONTRADA: {best_variant['variant']} > baseline por 5%")
            best_params = dict(BASELINE_PARAMS)
            for key in best_variant["mutations"]:
                best_params[key] = variants[int(best_variant["variant"].split("_")[1]) - 1][key]
            with open(PARAMS_PATH, "w") as f:
                json.dump({"params": best_params, "score": best_variant["score"],
                           "baseline_score": baseline_score}, f, indent=2)
            print(f"  Params guardados en: {PARAMS_PATH}")
        else:
            print(f"\n  Ninguna variante supera baseline por 5%")

    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "live_performance": live_perf,
        "alpha_decay": decay,
        "baseline_metrics": baseline_metrics,
        "variants_tested": len(results),
        "best_variant": best_variant,
        "all_variants": results,
        "recommendation": "consider_deploy" if (best_variant and baseline_metrics and
                                                 best_variant.get("score", 0) > baseline_metrics.get("cagr", 0) * 0.5 + baseline_metrics.get("win_rate", 0) * 0.3 + baseline_metrics.get("sharpe", 0) * 20 * 0.2) * 1.05
                          else "keep_baseline",
    }
    save_report(report)
    print(f"\n[6] Reporte guardado: {REPORT_PATH}")
    print(f"\n{'='*70}")
    print(f"AUTO-MEJORA COMPLETADA - Recomendacion: {report['recommendation']}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
