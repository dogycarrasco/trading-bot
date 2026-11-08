"""
Utilidades extraidas por ingenieria inversa de 4 repos open-source.

1. ROAI (Ghostfolio) - Return on Average Investment ponderado por dias
2. SEC EDGAR fetcher (OpenBB) - fundamentales sin API key
3. Modified Z-score (Metabase) - deteccion de outliers robusta
4. Volatilidad + Tendencia (Metabase time_series)
"""
import os
import sys
import time
import json
import threading
import requests
import numpy as np
import pandas as pd
from decimal import Decimal, getcontext

getcontext().prec = 28

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
CACHE_DIR = os.path.join(ROOT, "data", "external")
os.makedirs(CACHE_DIR, exist_ok=True)


# ============================================================
# 1. ROAI - Return on Average Investment (de Ghostfolio)
# ============================================================

def compute_roai(transactions, prices, end_date):
    """
    Calcula ROAI ponderado por dias, igual que Ghostfolio.

    transactions: [{date, type: 'BUY'|'SELL'|DIVIDEND, quantity, unit_price, fee}]
    prices: {date: close_price}
    end_date: fecha de evaluacion

    Devuelve: {roai_pct, net_performance, average_investment, total_investment_days}
    """
    if not transactions:
        return {"roai_pct": 0, "net_performance": 0, "average_investment": 0, "total_investment_days": 0}

    txs = sorted(transactions, key=lambda x: x["date"])
    initial_date = txs[0]["date"]

    sum_of_weighted_investments = Decimal("0")
    total_investment_days = 0
    cumulative_qty = Decimal("0")
    cumulative_investment = Decimal("0")
    last_date = initial_date

    for tx in txs:
        days_period = (tx["date"] - last_date).days
        if days_period > 0:
            sum_of_weighted_investments += cumulative_investment * Decimal(days_period)
            total_investment_days += days_period
        qty = Decimal(str(tx["quantity"]))
        price = Decimal(str(tx["unit_price"]))
        fee = Decimal(str(tx.get("fee", 0)))
        if tx["type"] == "BUY":
            cumulative_qty += qty
            cumulative_investment += qty * price + fee
        elif tx["type"] == "SELL":
            cumulative_qty -= qty
            cumulative_investment -= qty * price - fee
        elif tx["type"] == "DIVIDEND":
            cumulative_investment -= qty * price
        last_date = tx["date"]

    days_final = (end_date - last_date).days
    if days_final > 0:
        sum_of_weighted_investments += cumulative_investment * Decimal(days_final)
        total_investment_days += days_final

    if cumulative_investment > 0 and total_investment_days > 0:
        average_investment = (sum_of_weighted_investments + cumulative_investment) / Decimal(total_investment_days + 1)
    elif total_investment_days > 0:
        average_investment = sum_of_weighted_investments / Decimal(total_investment_days)
    else:
        average_investment = Decimal("0")

    final_price = Decimal(str(prices.get(end_date, 0)))
    market_value = cumulative_qty * final_price

    net_performance = market_value - cumulative_investment
    roai_pct = float(net_performance / average_investment * 100) if average_investment > 0 else 0

    return {
        "roai_pct": round(roai_pct, 4),
        "net_performance": float(net_performance),
        "average_investment": float(average_investment),
        "total_investment_days": total_investment_days,
        "cumulative_qty": float(cumulative_qty),
        "market_value": float(market_value),
    }


# ============================================================
# 2. SEC EDGAR fetcher (de OpenBB) - SIN API KEY
# ============================================================

class RateLimiter:
    """Token-bucket rate limiter thread-safe, puerto de OpenBB ratelimit.py"""
    _lock = threading.Lock()
    _next_available = 0.0

    @classmethod
    def wait(cls, requests_per_second=9.0):
        interval = 1.0 / requests_per_second
        with cls._lock:
            now = time.monotonic()
            scheduled = max(now, cls._next_available)
            cls._next_available = scheduled + interval
        wait = scheduled - now
        if wait > 0:
            time.sleep(wait)


_disk_cache = {}

def _cache_get(key):
    return _disk_cache.get(key)

def _cache_set(key, value, ttl=86400):
    _disk_cache[key] = {"value": value, "ts": time.time(), "ttl": ttl}

def _is_cache_valid(entry):
    return entry and (time.time() - entry["ts"]) < entry.get("ttl", 86400)


def sec_cik_map(symbol):
    """Ticker -> CIK number via SEC tickers.txt. Sin API key."""
    cache_key = f"cik_{symbol}"
    cached = _cache_get(cache_key)
    if _is_cache_valid(cached):
        return cached["value"]
    RateLimiter.wait(9)
    url = "https://www.sec.gov/files/company_tickers.json"
    headers = {"User-Agent": "TradingBot/1.0 (research@example.com)"}
    try:
        r = requests.get(url, headers=headers, timeout=15)
        if r.status_code == 200:
            data = r.json()
            for k, v in data.items():
                if v.get("ticker", "").upper() == symbol.upper():
                    cik = str(v["cik_str"])
                    _cache_set(cache_key, cik, ttl=86400 * 30)
                    return cik
    except Exception:
        pass
    return None


def sec_company_facts(symbol):
    """Obtiene financial statements de SEC EDGAR companyfacts JSON."""
    cache_key = f"facts_{symbol}"
    cached = _cache_get(cache_key)
    if _is_cache_valid(cached):
        return cached["value"]
    cik = sec_cik_map(symbol)
    if not cik:
        return {}
    cik_padded = cik.zfill(10)
    RateLimiter.wait(9)
    url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik_padded}.json"
    headers = {"User-Agent": "TradingBot/1.0 (research@example.com)"}
    try:
        r = requests.get(url, headers=headers, timeout=30)
        if r.status_code == 200:
            data = r.json()
            _cache_set(cache_key, data, ttl=86400 * 7)
            return data
    except Exception:
        pass
    return {}


def sec_fundamentals(symbol):
    """Extrae fundamentales clave de SEC companyfacts."""
    facts = sec_company_facts(symbol)
    if not facts or "facts" not in facts:
        return {}
    us_gaap = facts.get("facts", {}).get("us-gaap", {})

    def latest_value(concept, unit_key=None):
        if concept not in us_gaap:
            return None
        units = us_gaap[concept].get("units", {})
        if unit_key:
            data = units.get(unit_key, [])
        else:
            data = list(units.values())[0] if units else []
        if not data:
            return None
        latest = sorted(data, key=lambda x: x.get("end", ""), reverse=True)[0]
        return latest.get("val")

    return {
        "symbol": symbol,
        "revenue": latest_value("Revenues") or latest_value("RevenueFromContractWithCustomerExcludingAssessedTax"),
        "net_income": latest_value("NetIncomeLoss"),
        "eps": latest_value("EarningsPerShareBasic", "USD/shares"),
        "total_assets": latest_value("Assets"),
        "total_liabilities": latest_value("Liabilities"),
        "equity": latest_value("StockholdersEquity"),
        "current_assets": latest_value("AssetsCurrent"),
        "current_liabilities": latest_value("LiabilitiesCurrent"),
        "cash": latest_value("CashAndCashEquivalentsAtCarryingValue"),
        "debt": latest_value("LongTermDebt"),
    }


# ============================================================
# 3. Modified Z-score con MAD (de Metabase)
# ============================================================

def modified_zscore(values):
    """
    Modified Z-score usando MAD (Median Absolute Deviation).
    Mas robusto a outliers que el z-score clasico.
    """
    values = np.asarray(values, dtype=float)
    median = np.median(values)
    mad = np.median(np.abs(values - median))
    if mad == 0:
        return np.zeros_like(values)
    return 0.6745 * (values - median) / mad


def detect_anomalies(values, threshold=3.0):
    """Devuelve mascara booleana de outliers (|modified_z| > threshold)."""
    mz = modified_zscore(values)
    return np.abs(mz) > threshold


# ============================================================
# 4. Volatilidad + Tendencia (de Metabase time_series.clj)
# ============================================================

def compute_volatility_level(values):
    """
    Coeficiente de variacion = sigma / |mean|.
    Niveles: low (<0.1), moderate (<0.3), high (<0.5), extreme (>=0.5)
    """
    values = np.asarray(values, dtype=float)
    if len(values) < 2:
        return {"level": "unknown", "cv": 0}
    mean = np.mean(values)
    std = np.std(values)
    if mean == 0:
        return {"level": "unknown", "cv": 0}
    cv = std / abs(mean)
    if cv < 0.1:
        level = "low"
    elif cv < 0.3:
        level = "moderate"
    elif cv < 0.5:
        level = "high"
    else:
        level = "extreme"
    return {"level": level, "cv": float(cv), "max_change_pct": float((values.max() - values.min()) / mean * 100)}


def compute_trend(values):
    """Regresion lineal para detectar tendencia."""
    values = np.asarray(values, dtype=float)
    if len(values) < 3:
        return {"direction": "unknown", "slope": 0}
    x = np.arange(len(values))
    slope = np.polyfit(x, values, 1)[0]
    mean = np.mean(values)
    if mean == 0:
        return {"direction": "flat", "slope": float(slope)}
    pct_change = slope / mean * len(values) * 100
    if abs(pct_change) < 5:
        direction = "flat"
    elif pct_change > 0:
        direction = "up"
    else:
        direction = "down"
    return {"direction": direction, "slope": float(slope), "pct_change": float(pct_change)}