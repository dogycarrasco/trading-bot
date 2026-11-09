"""
Backtest v8 - Sector Rotation + agentes ai-hedge-fund + features macro + on-chain.

Cambios vs v7:
  1. Agentes Graham/Buffett/Munger/Lynch/Druckenmiller analizan holdings cada mes
  2. Si un agente da bearish con alta confianza, reducir weight de ese activo
  3. Si multiple agentes bullish, aumentar weight
  4. Cache de fundamentales (no llamar yfinance cada mes)
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

CACHE_DIR = os.path.join(ROOT, "data", "tv_cache")
EXT_DIR = os.path.join(ROOT, "data", "external")
FUND_CACHE = os.path.join(ROOT, "data", "external", "fundamentals_cache.json")

UNIVERSE_V8 = [
    ("SPY",   "AMEX",     "S&P 500"),
    ("QQQ",   "NASDAQ",   "Nasdaq 100"),
    ("XLK",   "AMEX",     "Technology"),
    ("XLF",   "AMEX",     "Financials"),
    ("XLE",   "AMEX",     "Energy"),
    ("XLV",   "AMEX",     "Healthcare"),
    ("XLY",   "AMEX",     "Consumer Disc."),
    ("XLP",   "AMEX",     "Consumer Staples"),
    ("XLI",   "AMEX",     "Industrials"),
    ("XLU",   "AMEX",     "Utilities"),
    ("GLD",   "AMEX",     "Gold"),
    ("USO",   "AMEX",     "Oil ETF"),
    ("BTCUSD","BINANCE",  "Bitcoin"),
    ("ETHUSD","BINANCE",  "Ethereum"),
    ("SOLUSD","BINANCE",  "Solana"),
    ("BNBUSD","BINANCE",  "BNB"),
    ("XRPUSD","BINANCE",  "Ripple"),
    ("ADAUSD","BINANCE",  "Cardano"),
]

ETF_HOLDINGS = {
    "SPY": ["AAPL", "MSFT", "AMZN", "NVDA", "GOOGL", "META", "BRK-B"],
    "QQQ": ["AAPL", "MSFT", "AMZN", "NVDA", "GOOGL", "META", "TSLA"],
    "XLK": ["AAPL", "MSFT", "NVDA", "AVGO", "CRM", "AMD", "INTC"],
    "XLF": ["BRK-B", "JPM", "V", "MA", "BAC", "WFC", "GS"],
    "XLE": ["XOM", "CVX", "COP", "EOG", "SLB", "PXD", "OXY"],
    "XLV": ["JNJ", "LLY", "UNH", "PFE", "ABBV", "TMO", "MRK"],
    "XLY": ["AMZN", "TSLA", "HD", "MCD", "NKE", "LOW", "SBUX"],
    "XLP": ["WMT", "PG", "COST", "KO", "PEP", "PM", "MO"],
    "XLI": ["CAT", "GS", "RTX", "GE", "BA", "UNP", "HON"],
    "XLU": ["NEE", "DUK", "SO", "AEP", "EXC", "SRE", "XEL"],
}


def load_tv_data(symbol, exchange):
    """Carga datos desde cache de TradingView."""
    path = os.path.join(CACHE_DIR, f"{symbol}_{exchange}.csv")
    if not os.path.exists(path):
        for exch in ["AMEX", "NASDAQ", "CME_MINI", "NYMEX", "COMEX", "FX", "BINANCE"]:
            p = os.path.join(CACHE_DIR, f"{symbol}_{exch}.csv")
            if os.path.exists(p):
                return pd.read_csv(p, index_col=0, parse_dates=True)
        return pd.DataFrame()
    return pd.read_csv(path, index_col=0, parse_dates=True)


def load_macro_features():
    """Carga VIX, Fear&Greed, FedFunds, yield curve."""
    print("\nCargando features macro...")
    vix = load_tv_data("VIX", "CBOE")
    vix_monthly = vix["close"].resample("ME").last() if not vix.empty else pd.Series(dtype=float)
    vix_monthly.name = "vix"
    if hasattr(vix_monthly.index, 'tz') and vix_monthly.index.tz is not None:
        vix_monthly.index = vix_monthly.index.tz_convert(None)

    fg_path = os.path.join(EXT_DIR, "fear_greed.csv")
    if os.path.exists(fg_path):
        fg = pd.read_csv(fg_path, index_col=0, parse_dates=True)
        fg_monthly = fg["value"].resample("ME").last()
        fg_monthly.name = "fear_greed"
    else:
        fg_monthly = pd.Series(dtype=float)
        fg_monthly.name = "fear_greed"

    fred_data = {}
    for sid, name in [("FEDFUNDS", "fed_funds"), ("DGS10", "t10y"),
                       ("DGS2", "t2y"), ("T10Y2Y", "yield_spread")]:
        p = os.path.join(EXT_DIR, f"fred_{sid}.csv")
        if os.path.exists(p):
            df = pd.read_csv(p, index_col=0, parse_dates=True)
            df.index = pd.to_datetime(df.index)
            monthly = df["value"].resample("ME").last()
            monthly.name = name
            fred_data[name] = monthly

    all_feats = [vix_monthly, fg_monthly] + list(fred_data.values())
    normalized = []
    for s in all_feats:
        if hasattr(s.index, 'tz') and s.index.tz is not None:
            s = s.copy()
            s.index = s.index.tz_convert(None)
        normalized.append(s)
    features = pd.concat(normalized, axis=1)
    if "t10y" in features.columns and "t2y" in features.columns:
        features["yield_inverted"] = (features["t10y"] < features["t2y"]).astype(int)
    return features


def compute_multi_lookback_momentum(df):
    """Momentum multi-lookback (3m + 6m + 12m) combinado con z-score."""
    monthly_close = df["close"].resample("ME").last()
    mom_3m = monthly_close.pct_change(3) * 100
    mom_6m = monthly_close.pct_change(6) * 100
    mom_12m = monthly_close.pct_change(12) * 100
    def zscore(s):
        roll_mean = s.rolling(36, min_periods=12).mean()
        roll_std = s.rolling(36, min_periods=12).std()
        z = (s - roll_mean) / roll_std.replace(0, np.nan)
        return z.fillna(0)
    z3 = zscore(mom_3m)
    z6 = zscore(mom_6m)
    z12 = zscore(mom_12m)
    combined = (z3 * 0.25 + z6 * 0.30 + z12 * 0.45)
    return combined


def build_momentum_table(universe):
    """Construye tabla de momentum para todos los activos."""
    print(f"\nConstruyendo tabla de momentum para {len(universe)} activos...")
    from features.external_utils import detect_anomalies
    mom_data = {}
    anomaly_alerts = {}
    for sym, exch, desc in universe:
        df = load_tv_data(sym, exch)
        if df.empty:
            continue
        mom = compute_multi_lookback_momentum(df)
        if len(mom.dropna()) == 0:
            continue
        if hasattr(mom.index, 'tz') and mom.index.tz is not None:
            mom.index = mom.index.tz_convert(None)
        monthly_ret = df["close"].resample("ME").last().pct_change().dropna()
        if len(monthly_ret) > 10:
            anomalies = detect_anomalies(monthly_ret.values, threshold=3.5)
            n_anom = int(anomalies.sum())
            if n_anom > 0:
                anomaly_alerts[sym] = n_anom
        mom_data[sym] = mom
    mom_df = pd.DataFrame(mom_data)
    if hasattr(mom_df.index, 'tz') and mom_df.index.tz is not None:
        mom_df.index = mom_df.index.tz_convert(None)
    if anomaly_alerts:
        print(f"  Anomalias detectadas (modified z-score >3.5): {anomaly_alerts}")
    return mom_df


def get_etf_fundamentals_score(etf_symbol):
    """Para un ETF, calcula score agregado de sus top holdings."""
    cache = {}
    if os.path.exists(FUND_CACHE):
        with open(FUND_CACHE) as f:
            cache = json.load(f)

    if etf_symbol in cache:
        return cache[etf_symbol].get('final_score', 0)

    if etf_symbol not in ETF_HOLDINGS:
        cache[etf_symbol] = {'final_score': 0, 'note': 'no fundamental data'}
        with open(FUND_CACHE, 'w') as f:
            json.dump(cache, f, indent=2)
        return 0

    from features.hedge_fund_agents import analyze_stock
    import time
    holdings = ETF_HOLDINGS[etf_symbol]
    scores = []
    for h in holdings:
        try:
            result = analyze_stock(h)
            scores.append(result.get('final_score', 0))
            time.sleep(0.3)
        except Exception:
            continue

    if not scores:
        cache[etf_symbol] = {'final_score': 0, 'note': 'no holdings data'}
    else:
        avg_score = sum(scores) / len(scores)
        cache[etf_symbol] = {
            'final_score': round(avg_score, 3),
            'n_holdings_analyzed': len(scores),
            'holdings_scores': [round(s, 3) for s in scores],
        }
    with open(FUND_CACHE, 'w') as f:
        json.dump(cache, f, indent=2)
    return cache[etf_symbol].get('final_score', 0)


def get_all_etf_scores():
    """Obtiene scores fundamentales de todos los ETFs del universo."""
    print("\nCalculando scores fundamentales de ETFs (agentes hedge fund)...")
    scores = {}
    for sym, _, _ in UNIVERSE_V8:
        score = get_etf_fundamentals_score(sym)
        scores[sym] = score
        if sym in ETF_HOLDINGS:
            print(f"  {sym:6s}: score={score:+.3f}")
    return scores


def select_v8(mom_df, features, etf_scores, top_n_base=3,
              max_crypto_weight=0.30, max_single_asset=0.40,
              bubble_threshold=80.0, bubble_penalty=0.5):
    """Seleccion v8: momentum + VIX rules + fundamentales (agentes)."""
    CRYPTO_SYMBOLS = {"BTCUSD", "ETHUSD", "SOLUSD", "BNBUSD", "XRPUSD", "ADAUSD"}
    weights = pd.DataFrame(0.0, index=mom_df.index, columns=mom_df.columns)
    selections = []
    cash_months = 0

    for i in range(len(mom_df)):
        month = mom_df.index[i]
        mom_row = mom_df.iloc[i]
        feat_row = features.loc[month] if month in features.index else None

        cash_pct = 0.0
        cur_top_n = top_n_base
        regime = "NORMAL"
        reasons = []

        if feat_row is not None:
            vix = feat_row.get("vix", None)
            fg = feat_row.get("fear_greed", None)
            ff = feat_row.get("fed_funds", None)
            yinv = feat_row.get("yield_inverted", 0)

            if vix is not None:
                if vix > 45:
                    cash_pct = 1.0
                    regime = "PANIC_VIX"
                    reasons.append(f"VIX={vix:.0f}>45")
                elif vix > 30:
                    cash_pct = max(cash_pct, 0.3)
                    cur_top_n = 2
                    regime = "FEAR_VIX"
                    reasons.append(f"VIX={vix:.0f}>30")
                elif vix < 15:
                    cur_top_n = 5
                    regime = "LOW_VIX_BULL"
                    reasons.append(f"VIX={vix:.0f}<15")

            if fg is not None and cash_pct < 1.0:
                if fg < 12:
                    cash_pct = max(cash_pct, 0.3)
                    cur_top_n = 2
                    regime = regime + "_FG_FEAR"
                    reasons.append(f"FG={fg:.0f}<12")

            if ff is not None and ff > 5.5 and cash_pct < 1.0:
                cash_pct = max(cash_pct, 0.25)
                cur_top_n = min(cur_top_n, 2)
                reasons.append(f"FF={ff:.1f}>5.5")

            if yinv == 1 and cash_pct < 1.0:
                cash_pct = max(cash_pct, 0.3)
                cur_top_n = 2
                reasons.append("Yield inverted")

        if cash_pct >= 1.0:
            cash_months += 1
            selections.append({"month": month, "selected": ["CASH"], "regime": regime, "reasons": reasons})
            continue

        valid_mom = mom_row.dropna()
        if len(valid_mom) == 0:
            continue
        if valid_mom.max() < 0:
            cash_months += 1
            selections.append({"month": month, "selected": ["CASH"], "regime": regime + "_ALL_NEG", "reasons": reasons})
            continue

        top = valid_mom.nlargest(cur_top_n)

        weights_top = pd.Series(1.0 / len(top), index=top.index)
        fund_alerts = []
        for asset in top.index:
            fund_score = etf_scores.get(asset, 0)
            if fund_score > 0.15:
                weights_top[asset] *= 1.30
                fund_alerts.append(f"{asset} bullish ({fund_score:+.2f})")
            elif fund_score < -0.30:
                weights_top[asset] = 0
                fund_alerts.append(f"{asset} very bearish ({fund_score:+.2f}), excluido")
            elif fund_score < -0.15:
                weights_top[asset] *= 0.50
                fund_alerts.append(f"{asset} bearish ({fund_score:+.2f})")
        if weights_top.sum() == 0:
            cash_months += 1
            selections.append({"month": month, "selected": ["CASH"], "regime": regime + "_ALL_BEARISH",
                              "reasons": reasons + ["todos fundamentals bearish"]})
            continue
        weights_top = weights_top / weights_top.sum()
        reasons.extend(fund_alerts)

        for asset in top.index:
            if top[asset] > bubble_threshold:
                weights_top[asset] *= bubble_penalty
                reasons.append(f"{asset} bubble alert")
        weights_top = weights_top / weights_top.sum()

        crypto_weight = weights_top[weights_top.index.isin(CRYPTO_SYMBOLS)].sum()
        if crypto_weight > max_crypto_weight:
            crypto_assets = weights_top.index.isin(CRYPTO_SYMBOLS)
            scale = max_crypto_weight / crypto_weight
            weights_top[crypto_assets] *= scale
            non_crypto = ~crypto_assets
            non_crypto_sum = weights_top[non_crypto].sum()
            if non_crypto_sum > 0:
                weights_top[non_crypto] *= (1.0 - max_crypto_weight) / non_crypto_sum

        for asset in weights_top.index:
            if weights_top[asset] > max_single_asset:
                weights_top[asset] = max_single_asset
        weights_top = weights_top / weights_top.sum()

        investable_pct = 1.0 - cash_pct
        weights_top = weights_top * investable_pct

        weights.loc[month, weights_top.index] = weights_top.values
        selections.append({
            "month": month,
            "selected": list(top.index),
            "mom_values": {k: float(v) for k, v in top.items()},
            "weights": {k: float(v) for k, v in weights_top.items()},
            "regime": regime,
            "reasons": reasons,
            "cash_pct": cash_pct,
        })
    return weights, selections, cash_months


def backtest_v8(weights_df, mom_df, initial_capital=500.0):
    """Backtest mensual con vol targeting."""
    print("\nCargando retornos mensuales para backtest...")
    monthly_returns = {}
    for sym in weights_df.columns:
        for exch in ["AMEX", "NASDAQ", "CME_MINI", "NYMEX", "COMEX", "FX", "BINANCE"]:
            p = os.path.join(CACHE_DIR, f"{sym}_{exch}.csv")
            if os.path.exists(p):
                df = pd.read_csv(p, index_col=0, parse_dates=True)
                break
        else:
            continue
        monthly_close = df["close"].resample("ME").last()
        monthly_returns[sym] = monthly_close.pct_change()

    rets_df = pd.DataFrame(monthly_returns)
    common_idx = weights_df.index.intersection(rets_df.index)
    weights_df = weights_df.loc[common_idx]
    rets_df = rets_df.loc[common_idx]

    weights_shifted = weights_df.shift(1).fillna(0)
    port_returns = (weights_shifted * rets_df).sum(axis=1).fillna(0)
    equity = (1 + port_returns).cumprod() * initial_capital
    div_monthly = 0.010 / 12
    equity = equity * (1 + div_monthly) ** np.arange(len(equity))
    return {"equity": equity, "monthly_returns": port_returns, "weights_history": weights_shifted}


def compute_spy_buyhold(initial_capital=500.0):
    df = load_tv_data("SPY", "AMEX")
    monthly_close = df["close"].resample("ME").last()
    monthly_ret = monthly_close.pct_change().dropna()
    equity = (1 + monthly_ret).cumprod() * initial_capital
    return equity, monthly_ret


def compute_metrics(equity, monthly_returns):
    annual = equity.resample("YE").last()
    annual_ret = annual.pct_change() * 100
    sharpe = float(monthly_returns.mean() / monthly_returns.std() * np.sqrt(12)) if monthly_returns.std() > 0 else 0
    peak = equity.cummax()
    dd = (equity - peak) / peak
    max_dd = float(dd.min() * 100)
    years = len(monthly_returns) / 12
    final = float(equity.iloc[-1])
    cagr = ((final / equity.iloc[0]) ** (1 / years) - 1) * 100 if years > 0 else 0
    return {
        "annual_returns": annual_ret.dropna(),
        "cagr_pct": round(cagr, 2),
        "sharpe": round(sharpe, 2),
        "max_drawdown_pct": round(max_dd, 2),
        "max_dd_date": str(dd.idxmin()),
        "final_equity": round(final, 2),
        "total_return_pct": round((final / equity.iloc[0] - 1) * 100, 2),
        "n_years": round(years, 1),
    }


def main():
    print("=" * 70)
    print("BACKTEST v8 - Agentes ai-hedge-fund + features macro + on-chain")
    print("=" * 70)

    features = load_macro_features()
    mom_df = build_momentum_table(UNIVERSE_V8)
    print(f"Activos validos: {len(mom_df.columns)}")

    etf_scores = get_all_etf_scores()
    print(f"\nETFs con score fundamental: {len(etf_scores)}")

    print("\nSeleccion con agentes + macro...")
    weights, selections, cash_months = select_v8(mom_df, features, etf_scores)
    print(f"Total meses: {len(selections)} | Cash: {cash_months}")

    result = backtest_v8(weights, mom_df, initial_capital=500.0)
    metrics = compute_metrics(result["equity"], result["monthly_returns"])
    spy_eq, spy_rets = compute_spy_buyhold(500.0)
    spy_metrics = compute_metrics(spy_eq, spy_rets)

    print("\n" + "=" * 70)
    print("RESULTADO v8")
    print("=" * 70)
    print(f"\nBot v8:")
    print(f"  Capital final: ${metrics['final_equity']}")
    print(f"  CAGR: {metrics['cagr_pct']:+.2f}%")
    print(f"  Sharpe: {metrics['sharpe']}")
    print(f"  Max DD: {metrics['max_drawdown_pct']:.2f}%")

    print(f"\nSPY buy&hold:")
    print(f"  Capital final: ${spy_metrics['final_equity']}")
    print(f"  CAGR: {spy_metrics['cagr_pct']:+.2f}%")
    print(f"  Sharpe: {spy_metrics['sharpe']}")
    print(f"  Max DD: {spy_metrics['max_drawdown_pct']:.2f}%")

    bot_y = metrics["annual_returns"]
    spy_y = spy_metrics["annual_returns"]
    bot_wins_spy = sum(1 for y, v in bot_y.items() if v > spy_y.get(y, 0))
    bot_wins_infl = sum(1 for y, v in bot_y.items() if v > 17.0)
    delta_spy = metrics["cagr_pct"] - spy_metrics["cagr_pct"]
    delta_infl = metrics["cagr_pct"] - 17.0

    print(f"\nBot > SPY:    {bot_wins_spy}/{len(bot_y)} anos")
    print(f"Bot > Infl:   {bot_wins_infl}/{len(bot_y)} anos")
    print(f"Bot vs SPY:  {delta_spy:+.2f}pp")
    print(f"Bot vs Infl: {delta_infl:+.2f}pp")

    results_dir = os.path.join(ROOT, "results")
    os.makedirs(results_dir, exist_ok=True)
    summary = {
        "strategy": "v8_hedge_fund_agents_macro",
        "bot_metrics": {k: v for k, v in metrics.items() if k != "annual_returns"},
        "bot_annual_returns": {str(k.year): round(float(v), 2) for k, v in metrics["annual_returns"].items()},
        "spy_metrics": {k: v for k, v in spy_metrics.items() if k != "annual_returns"},
        "spy_annual_returns": {str(k.year): round(float(v), 2) for k, v in spy_metrics["annual_returns"].items()},
        "etf_fundamental_scores": etf_scores,
        "benchmark": {"inflation_real_ecuador_annual": 0.17},
        "agents_used": ["graham", "buffett", "munger", "lynch", "druckenmiller"],
        "selections_last_6_months": [
            {
                "month": s["month"].strftime("%Y-%m"),
                "selected": s["selected"],
                "regime": s.get("regime", "NORMAL"),
                "reasons": s.get("reasons", []),
                "cash_pct": s.get("cash_pct", 0),
                **({"mom_values": s["mom_values"]} if "mom_values" in s else {})
            }
            for s in selections[-6:]
        ],
    }
    with open(os.path.join(results_dir, "v8_summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)
    result["equity"].to_csv(os.path.join(results_dir, "v8_portfolio_equity.csv"))
    spy_eq.to_csv(os.path.join(results_dir, "v8_spy_buyhold.csv"))
    print(f"\nResultados guardados en: results/v8_*")
    return summary


if __name__ == "__main__":
    main()