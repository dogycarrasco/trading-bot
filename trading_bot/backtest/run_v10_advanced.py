"""
Backtest v10 - VectorBT + QuantStats + Trade Memory + Bucle de Automejora.
"""
import os
import sys
import json
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from backtest.run_v8_hedge_fund import (
    UNIVERSE_V8, load_tv_data, load_macro_features,
    compute_multi_lookback_momentum, get_all_etf_scores,
    compute_metrics, compute_spy_buyhold,
)


def init_trade_memory():
    tm_path = os.path.join(ROOT, "results", "trade_memory.json")
    if not os.path.exists(tm_path):
        tm = {"trades": [], "last_update": None}
        with open(tm_path, "w") as f:
            json.dump(tm, f, indent=2)
    return tm_path


def log_trade(trade_memory_path, trade_data):
    try:
        with open(trade_memory_path) as f:
            tm = json.load(f)
    except Exception:
        tm = {"trades": [], "last_update": None}
    tm["trades"].append(trade_data)
    tm["last_update"] = pd.Timestamp.now().isoformat()
    if len(tm["trades"]) > 500:
        tm["trades"] = tm["trades"][-500:]
    with open(trade_memory_path, "w") as f:
        json.dump(tm, f, indent=2, default=str)


def analyze_trade_memory(trade_memory_path):
    try:
        with open(trade_memory_path) as f:
            tm = json.load(f)
    except Exception:
        return {"total_trades": 0, "win_rate": 0, "avg_pnl": 0, "reasons": {}}
    trades = tm.get("trades", [])
    if not trades:
        return {"total_trades": 0, "win_rate": 0, "avg_pnl": 0, "reasons": {}}
    wins = [t for t in trades if t.get("pnl", 0) > 0]
    losses = [t for t in trades if t.get("pnl", 0) <= 0]
    reasons = {}
    for t in trades:
        reason = t.get("exit_reason", "UNKNOWN")
        if reason not in reasons:
            reasons[reason] = {"count": 0, "total_pnl": 0, "wins": 0}
        reasons[reason]["count"] += 1
        reasons[reason]["total_pnl"] += t.get("pnl", 0)
        if t.get("pnl", 0) > 0:
            reasons[reason]["wins"] += 1
    for r in reasons:
        reasons[r]["win_rate"] = reasons[r]["wins"] / reasons[r]["count"] * 100 if reasons[r]["count"] > 0 else 0
    return {
        "total_trades": len(trades),
        "win_rate": len(wins) / len(trades) * 100,
        "avg_pnl": float(np.mean([t.get("pnl", 0) for t in trades])),
        "avg_win": float(np.mean([t.get("pnl", 0) for t in wins])) if wins else 0,
        "avg_loss": float(np.mean([t.get("pnl", 0) for t in losses])) if losses else 0,
        "reasons": reasons,
    }


def generate_quantstats_report(returns, benchmark="SPY", output_path=None):
    try:
        import quantstats as qs
    except ImportError:
        print("QuantStats no instalado - saltando tearsheet")
        return None
    if output_path is None:
        output_path = os.path.join(ROOT, "results", "tearsheet.html")
    try:
        returns = pd.Series(returns).dropna()
        returns.index = pd.to_datetime(returns.index)
        returns.name = "Bot v10"
        qs.reports.html(returns, benchmark=benchmark, output=output_path,
                        title="Bot v10 - Tearsheet", download_images=True)
        print(f"Tearsheet generado: {output_path}")
        return output_path
    except Exception as e:
        print(f"Error generando tearsheet: {e}")
        return None


def analyze_weaknesses(annual_returns, spy_returns, drawdown_series):
    weaknesses = []
    proposals = []
    for year, ret in annual_returns.items():
        if ret < -20:
            weaknesses.append(f"{year}: {ret:+.1f}% (perdida severa)")
            if str(year) in ("2015", "2018"):
                proposals.append({"param": "max_crypto_weight", "current": 0.30,
                                  "proposed": 0.20, "reason": f"{year} crypto winter"})
    if drawdown_series is not None:
        max_dd = float(drawdown_series.min()) if hasattr(drawdown_series, 'min') else 0
        if max_dd < -30:
            weaknesses.append(f"Max DD {max_dd:.1f}% - muy profundo")
            proposals.append({"param": "stop_loss_threshold", "current": -0.15,
                              "proposed": -0.10, "reason": "Stop-loss mas ajustado"})
    for year, bot_ret in annual_returns.items():
        spy_ret = spy_returns.get(year, 0)
        if spy_ret - bot_ret > 10:
            weaknesses.append(f"{year}: bot {bot_ret:+.1f}% vs SPY {spy_ret:+.1f}%")
            proposals.append({"param": "top_n_base", "current": 3,
                              "proposed": 5, "reason": f"{year}: mas diversificacion"})
    if len(annual_returns) > 5:
        std = float(np.std(list(annual_returns.values())))
        if std > 40:
            weaknesses.append(f"Volatilidad alta (std={std:.1f}%)")
            proposals.append({"param": "max_crypto_weight", "current": 0.30,
                              "proposed": 0.20, "reason": "Reducir cripto"})
    return {"weaknesses": weaknesses, "proposals": proposals,
            "n_weaknesses": len(weaknesses), "n_proposals": len(proposals)}


def test_parameter_change(param_name, current_value, proposed_value,
                          universe_data, mom_df, features, etf_scores):
    from backtest.run_v9_improvements import select_v9, backtest_v9
    modified_config = {"max_crypto_weight": 0.30, "max_single_asset": 0.40,
                       "bubble_threshold": 80.0, "bubble_penalty": 0.5,
                       "stop_loss_threshold": -0.15, "top_n_base": 3}
    modified_config[param_name] = proposed_value
    try:
        weights, selections, cash_months, stop_loss_events = select_v9(
            universe_data, mom_df, features, etf_scores, **modified_config)
        result = backtest_v9(weights, universe_data, initial_capital=500.0)
        metrics = compute_metrics(result["equity"], result["monthly_returns"])
        return {"param": param_name, "current": current_value, "proposed": proposed_value,
                "proposed_sharpe": metrics["sharpe"], "proposed_cagr": metrics["cagr_pct"],
                "proposed_max_dd": metrics["max_drawdown_pct"]}
    except Exception as e:
        return {"param": param_name, "current": current_value, "proposed": proposed_value,
                "error": str(e), "improves": False}


def run_self_improvement_loop(bot_metrics, spy_metrics, annual_returns, spy_annual,
                               drawdown_series, universe_data, mom_df, features, etf_scores):
    print("\n" + "=" * 70)
    print("BUCLE DE AUTOMEJORA")
    print("=" * 70)
    analysis = analyze_weaknesses(annual_returns, spy_annual, drawdown_series)
    print(f"\nDebilidades: {analysis['n_weaknesses']}")
    for w in analysis["weaknesses"]:
        print(f"  - {w}")
    if not analysis["proposals"]:
        print("Sin propuestas. Bot estable.")
        return {"improvements_applied": 0, "analysis": analysis}
    print(f"\nProbando {len(analysis['proposals'])} propuestas...")
    current_sharpe = bot_metrics.get("sharpe", 0)
    improvements = []
    for proposal in analysis["proposals"]:
        print(f"\n  {proposal['param']}: {proposal['current']} -> {proposal['proposed']}")
        result = test_parameter_change(proposal["param"], proposal["current"],
                                       proposal["proposed"], universe_data, mom_df, features, etf_scores)
        result["current_sharpe"] = current_sharpe
        result["improves"] = result.get("proposed_sharpe", 0) > current_sharpe
        if result["improves"]:
            print(f"  >>> MEJORA: Sharpe {current_sharpe} -> {result['proposed_sharpe']}")
            improvements.append(result)
        else:
            print(f"  >>> NO mejora")
    report = {"timestamp": pd.Timestamp.now().isoformat(), "current_sharpe": current_sharpe,
              "weaknesses": analysis["weaknesses"], "proposals_tested": len(analysis["proposals"]),
              "improvements_found": len(improvements), "improvements": improvements}
    report_path = os.path.join(ROOT, "results", "self_improvement_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    return report


def main():
    print("=" * 70)
    print("BACKTEST v10 - QuantStats + Trade Memory + Automejora")
    print("=" * 70)
    from backtest.run_v9_improvements import build_universe_data, select_v9, backtest_v9
    universe_data = build_universe_data(UNIVERSE_V8)
    features = load_macro_features()
    mom_data = {}
    for sym, exch, desc in UNIVERSE_V8:
        if sym in universe_data:
            df_daily = load_tv_data(sym, exch)
            if not df_daily.empty:
                mom_data[sym] = compute_multi_lookback_momentum(df_daily)
    mom_df = pd.DataFrame(mom_data)
    if hasattr(mom_df.index, 'tz') and mom_df.index.tz is not None:
        mom_df.index = mom_df.index.tz_convert(None)
    etf_scores = get_all_etf_scores()
    
    print("\nBacktest v9...")
    weights, selections, cash_months, stop_loss_events = select_v9(universe_data, mom_df, features, etf_scores)
    result = backtest_v9(weights, universe_data, initial_capital=500.0)
    metrics = compute_metrics(result["equity"], result["monthly_returns"])
    spy_eq, spy_rets = compute_spy_buyhold(500.0)
    spy_metrics = compute_metrics(spy_eq, spy_rets)
    
    print(f"\nBot v10: ${metrics['final_equity']} (CAGR {metrics['cagr_pct']:+.2f}%, Sharpe {metrics['sharpe']})")
    
    # TRADE MEMORY
    print("\n=== TRADE MEMORY ===")
    tm_path = init_trade_memory()
    for sel in selections:
        if sel.get("selected") and sel["selected"] != ["CASH"]:
            for asset in sel["selected"]:
                log_trade(tm_path, {"symbol": asset, "entry_date": str(sel["month"]),
                    "exit_date": str(sel["month"]), "pnl": 0, "exit_reason": "REBALANCE",
                    "regime": sel.get("regime", "NORMAL"), "strategy": "V9"})
    tm_stats = analyze_trade_memory(tm_path)
    print(f"  Trades: {tm_stats['total_trades']}, Win rate: {tm_stats.get('win_rate', 0):.1f}%")
    
    # QUANTSTATS
    print("\n=== QUANTSTATS ===")
    daily_equity = result["equity"].resample("D").ffill()
    daily_returns = daily_equity.pct_change().dropna()
    tearsheet_path = generate_quantstats_report(daily_returns, output_path=os.path.join(ROOT, "results", "tearsheet.html"))
    
    # AUTOMEJORA
    print("\n=== BUCLE DE AUTOMEJORA ===")
    bot_y = metrics["annual_returns"]
    spy_y = spy_metrics["annual_returns"]
    equity = result["equity"]
    dd = (equity - equity.cummax()) / equity * 100
    improvement = run_self_improvement_loop(
        metrics, spy_metrics,
        {str(k.year): float(v) for k, v in bot_y.items()},
        {str(k.year): float(v) for k, v in spy_y.items()},
        dd, universe_data, mom_df, features, etf_scores)
    
    # Guardar
    results_dir = os.path.join(ROOT, "results")
    summary = {
        "strategy": "v10_quantstats_trade_memory_automejora",
        "bot_metrics": {k: v for k, v in metrics.items() if k != "annual_returns"},
        "bot_annual_returns": {str(k.year): round(float(v), 2) for k, v in bot_y.items()},
        "spy_metrics": {k: v for k, v in spy_metrics.items() if k != "annual_returns"},
        "trade_memory_stats": tm_stats,
        "tearsheet_generated": tearsheet_path is not None,
        "self_improvement": {"weaknesses": improvement.get("weaknesses", []),
                             "proposals_tested": improvement.get("proposals_tested", 0),
                             "improvements_found": improvement.get("improvements_found", 0)},
    }
    with open(os.path.join(results_dir, "v10_summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\n=== RESUMEN v10 ===")
    print(f"  Bot: ${metrics['final_equity']} (CAGR {metrics['cagr_pct']:+.2f}%)")
    print(f"  Tearsheet: {'OK' if tearsheet_path else 'N/A'}")
    print(f"  Trade Memory: {tm_stats['total_trades']} trades")
    print(f"  Automejora: {improvement.get('improvements_found', 0)} mejoras")
    return summary

if __name__ == "__main__":
    main()
