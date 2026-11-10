"""
Simulador de opciones (hedging) + Analisis de tendencia bull/bear.
"""
import math
import numpy as np
import pandas as pd
from datetime import datetime, timedelta


def black_scholes_put(S, K, T, r, sigma):
    if T <= 0 or sigma <= 0 or S <= 0:
        return max(K - S, 0)
    try:
        from scipy.stats import norm
        d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
        d2 = d1 - sigma * math.sqrt(T)
        put_price = K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        return max(put_price, 0)
    except Exception:
        return max(K - S, 0)


def black_scholes_greeks(S, K, T, r, sigma):
    if T <= 0 or sigma <= 0 or S <= 0:
        return {"delta": -1.0, "gamma": 0, "theta": 0, "vega": 0}
    try:
        from scipy.stats import norm
        d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
        d2 = d1 - sigma * math.sqrt(T)
        delta = norm.cdf(d1) - 1
        gamma = norm.pdf(d1) / (S * sigma * math.sqrt(T))
        theta = (-(S * norm.pdf(d1) * sigma) / (2 * math.sqrt(T)) + r * K * math.exp(-r * T) * norm.cdf(-d2)) / 365
        vega = S * norm.pdf(d1) * math.sqrt(T) / 100
        return {"delta": delta, "gamma": gamma, "theta": theta, "vega": vega}
    except Exception:
        return {"delta": -1.0, "gamma": 0, "theta": 0, "vega": 0}


class OptionsHedgeManager:
    def __init__(self):
        self.positions = []
        self.history = []
        self.risk_free_rate = 0.045

    def should_hedge(self, market_score):
        return market_score < -20

    def estimate_implied_vol(self, vix_value):
        if vix_value and vix_value > 0:
            return vix_value / 100
        return 0.20

    def buy_put(self, underlying_price, vix, quantity=1):
        strike = underlying_price * 0.95
        days_to_expiry = 30
        T = days_to_expiry / 365
        sigma = self.estimate_implied_vol(vix)
        put_price = black_scholes_put(underlying_price, strike, T, self.risk_free_rate, sigma)
        greeks = black_scholes_greeks(underlying_price, strike, T, self.risk_free_rate, sigma)
        position = {
            "type": "PUT", "underlying": "SPY",
            "entry_date": datetime.now().isoformat(),
            "expiry_date": (datetime.now() + timedelta(days=days_to_expiry)).isoformat(),
            "strike": round(strike, 2), "entry_price": round(put_price, 2),
            "underlying_entry": underlying_price, "quantity": quantity,
            "contract_size": 100, "cost": round(put_price * quantity * 100, 2),
            "implied_vol": round(sigma * 100, 1), "days_to_expiry": days_to_expiry,
            "greeks_at_entry": greeks, "T": T,
        }
        self.positions.append(position)
        return position

    def update_positions(self, current_spy_price, vix):
        for pos in self.positions:
            days_left = (datetime.fromisoformat(pos["expiry_date"]) - datetime.now()).days
            if days_left <= 0:
                self._close_position(pos, current_spy_price, expired=True)
                continue
            T = days_left / 365
            sigma = self.estimate_implied_vol(vix)
            current_price = black_scholes_put(current_spy_price, pos["strike"], T, self.risk_free_rate, sigma)
            pos["current_price"] = round(current_price, 2)
            pos["current_value"] = round(current_price * pos["quantity"] * pos["contract_size"], 2)
            pos["pnl"] = round((current_price - pos["entry_price"]) * pos["quantity"] * pos["contract_size"], 2)
            pos["pnl_pct"] = round(((current_price / pos["entry_price"]) - 1) * 100, 1) if pos["entry_price"] > 0 else 0
            pos["days_to_expiry"] = days_left
            pos["current_greeks"] = black_scholes_greeks(current_spy_price, pos["strike"], T, self.risk_free_rate, sigma)
        self.positions = [p for p in self.positions if p.get("days_to_expiry", 0) > 0]

    def _close_position(self, pos, current_price, expired=False):
        if expired:
            exit_price = max(pos["strike"] - current_price, 0)
        else:
            exit_price = pos.get("current_price", 0)
        pnl = (exit_price - pos["entry_price"]) * pos["quantity"] * pos["contract_size"]
        self.history.append({**pos, "exit_date": datetime.now().isoformat(), "exit_price": round(exit_price, 2), "pnl": round(pnl, 2), "expired": expired})

    def close_all(self, current_spy_price, vix):
        self.update_positions(current_spy_price, vix)
        for pos in self.positions:
            self._close_position(pos, current_spy_price, expired=False)
        self.positions = []

    def total_hedge_value(self):
        return sum(p.get("current_value", p.get("cost", 0)) for p in self.positions)

    def total_hedge_cost(self):
        return sum(p["cost"] for p in self.positions)

    def total_hedge_pnl(self):
        return sum(p.get("pnl", 0) for p in self.positions)

    def get_summary(self):
        return {
            "open_positions": len(self.positions),
            "total_cost": self.total_hedge_cost(),
            "current_value": self.total_hedge_value(),
            "pnl": self.total_hedge_pnl(),
            "closed_positions": len(self.history),
            "history_pnl": sum(h.get("pnl", 0) for h in self.history),
        }


def compute_market_score(vix, fear_greed, spy_above_ema200, yield_inverted, fed_funds, spy_momentum_20d, spy_momentum_50d):
    score = 0
    if vix is not None:
        if vix < 15: score += 25
        elif vix < 20: score += 15
        elif vix < 25: score += 5
        elif vix < 30: score -= 5
        elif vix < 40: score -= 15
        else: score -= 25
    if fear_greed is not None:
        if fear_greed < 20: score -= 20
        elif fear_greed < 40: score -= 10
        elif fear_greed < 60: score += 0
        elif fear_greed < 80: score += 10
        else: score += 20
    if spy_above_ema200 is not None:
        score += 20 if spy_above_ema200 else -20
    if yield_inverted is not None:
        score -= 10 if yield_inverted else 5
    if fed_funds is not None:
        if fed_funds > 5.5: score -= 10
        elif fed_funds > 3.0: score += 0
        else: score += 10
    if spy_momentum_20d is not None:
        if spy_momentum_20d > 5: score += 10
        elif spy_momentum_20d > 0: score += 5
        elif spy_momentum_20d > -5: score -= 5
        else: score -= 10
    if spy_momentum_50d is not None:
        if spy_momentum_50d > 10: score += 5
        elif spy_momentum_50d > 0: score += 3
        else: score -= 5
    return max(-100, min(100, score))


def score_to_regime(score):
    if score >= 50: return "STRONG_BULL"
    elif score >= 20: return "BULL"
    elif score >= -20: return "NEUTRAL"
    elif score >= -50: return "BEAR"
    else: return "STRONG_BEAR"


def score_to_allocation(score):
    if score >= 50: return (1.0, 0.0, 0.0)
    elif score >= 20: return (0.90, 0.10, 0.0)
    elif score >= 0: return (0.70, 0.25, 0.05)
    elif score >= -20: return (0.50, 0.35, 0.15)
    elif score >= -50: return (0.30, 0.40, 0.30)
    else: return (0.10, 0.40, 0.50)
