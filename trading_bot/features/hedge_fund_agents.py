"""
Agentes de ai-hedge-fund aplicados como heuristicas cuantitativas.

En vez de usar LLM (requiere API key), implemento las reglas explicitas
de cada inversor legendario tal como aparecen en sus prompts del repo.

Fuentes:
  - ai-hedge-fund/signals/graham.py
  - buffett.py, munger.py, lynch.py, drunkenmiller.py

Cada agente devuelve: signal (bullish/bearish/neutral), confidence (0-100).
El portafolio combina los 5 con pesos (Graham x2 en deep-value strategy).
"""
import os
import json
import time
import requests
import pandas as pd
import numpy as np


def get_fundamentals(symbol):
    """Obtiene fundamentales via SEC EDGAR (sin API key, datos oficiales SEC)."""
    try:
        from features.external_utils import sec_fundamentals
        sec_data = sec_fundamentals(symbol)
        if not sec_data or not sec_data.get('revenue'):
            return _get_fundamentals_yfinance(symbol)

        revenue = sec_data.get('revenue', 0) or 0
        net_income = sec_data.get('net_income', 0) or 0
        equity = sec_data.get('equity', 0) or 0
        total_assets = sec_data.get('total_assets', 0) or 0
        current_assets = sec_data.get('current_assets', 0) or 0
        current_liabilities = sec_data.get('current_liabilities', 0) or 0
        debt = sec_data.get('debt', 0) or 0
        eps = sec_data.get('eps', 0) or 0

        profit_margin = net_income / revenue if revenue > 0 else 0
        roe = net_income / equity if equity > 0 else 0
        roa = net_income / total_assets if total_assets > 0 else 0
        current_ratio = current_assets / current_liabilities if current_liabilities > 0 else 0
        debt_to_equity = (debt / equity * 100) if equity > 0 else 999

        return {
            'symbol': symbol,
            'pe_ratio': 0,
            'peg_ratio': 0,
            'eps': eps,
            'book_value': equity / 1e9 if equity else 0,
            'dividend_yield': 0,
            'roe': roe,
            'roa': roa,
            'profit_margin': profit_margin,
            'revenue_growth': 0,
            'debt_to_equity': debt_to_equity,
            'current_ratio': current_ratio,
            'market_cap': 0,
            'sector': '',
            'industry': '',
            'source': 'SEC EDGAR',
            'revenue': revenue,
            'net_income': net_income,
            'total_assets': total_assets,
            'equity': equity,
        }
    except Exception:
        return _get_fundamentals_yfinance(symbol)


def _get_fundamentals_yfinance(symbol):
    """Fallback a yfinance si SEC falla."""
    try:
        import yfinance as yf
        ticker = yf.Ticker(symbol)
        info = ticker.info
        if not info or len(info) < 5:
            return {}
        return {
            'symbol': symbol,
            'pe_ratio': float(info.get('trailingPE', 0) or 0),
            'peg_ratio': float(info.get('pegRatio', 0) or 0),
            'eps': float(info.get('trailingEps', 0) or 0),
            'book_value': float(info.get('bookValue', 0) or 0),
            'dividend_yield': float(info.get('dividendYield', 0) or 0),
            'roe': float(info.get('returnOnEquity', 0) or 0),
            'roa': float(info.get('returnOnAssets', 0) or 0),
            'profit_margin': float(info.get('profitMargins', 0) or 0),
            'revenue_growth': float(info.get('revenueGrowth', 0) or 0),
            'debt_to_equity': float(info.get('debtToEquity', 0) or 0),
            'current_ratio': float(info.get('currentRatio', 0) or 0),
            'market_cap': float(info.get('marketCap', 0) or 0),
            'sector': info.get('sector', ''),
            'industry': info.get('industry', ''),
            'source': 'yfinance',
        }
    except Exception:
        return {}


def graham_agent(fund):
    """Benjamin Graham - margin of safety above all."""
    if not fund or not fund.get('pe_ratio'):
        return {'agent': 'graham', 'signal': 'neutral', 'confidence': 30, 'reasoning': 'Sin datos suficientes'}

    pe = fund['pe_ratio']
    cr = fund.get('current_ratio', 0)
    eps = fund.get('eps', 0)
    score = 0
    reasons = []

    if pe > 0 and pe < 15:
        score += 25
        reasons.append(f'P/E {pe:.1f} < 15 (margin of safety)')
    elif pe < 20:
        score += 10
    else:
        reasons.append(f'P/E {pe:.1f} alto')

    if cr > 1.5:
        score += 20
        reasons.append(f'Current ratio {cr:.2f} > 1.5')
    elif cr > 1.0:
        score += 10

    if eps > 0:
        score += 15
    else:
        reasons.append(f'EPS negativo {eps:.2f}')

    if fund.get('dividend_yield', 0) > 0.01:
        score += 10

    de = fund.get('debt_to_equity', 100)
    if de < 50:
        score += 15
    elif de > 100:
        score -= 10

    signal = 'bullish' if score >= 70 else ('bearish' if score <= 30 else 'neutral')
    return {'agent': 'graham', 'signal': signal, 'confidence': min(100, max(0, score)), 'reasoning': '; '.join(reasons)}


def buffett_agent(fund):
    """Warren Buffett - quality at reasonable price, moat, consistency."""
    if not fund or not fund.get('roe'):
        return {'agent': 'buffett', 'signal': 'neutral', 'confidence': 30, 'reasoning': 'Sin datos suficientes'}

    roe = fund['roe']
    pm = fund.get('profit_margin', 0)
    de = fund.get('debt_to_equity', 100)
    pe = fund.get('pe_ratio', 100)
    rg = fund.get('revenue_growth', 0)
    score = 0
    reasons = []

    if roe > 0.20:
        score += 30
        reasons.append(f'ROE {roe*100:.1f}% > 20% (moat claro)')
    elif roe > 0.15:
        score += 20
    else:
        reasons.append(f'ROE {roe*100:.1f}% bajo')

    if pm > 0.15:
        score += 20
    elif pm > 0.10:
        score += 10

    if de < 50:
        score += 15
    elif de > 100:
        score -= 10

    if pe > 0 and pe < 25:
        score += 15
    elif pe > 35:
        score -= 10

    if rg > 0.10:
        score += 20
    elif rg > 0:
        score += 10

    signal = 'bullish' if score >= 70 else ('bearish' if score <= 30 else 'neutral')
    return {'agent': 'buffett', 'signal': signal, 'confidence': min(100, max(0, score)), 'reasoning': '; '.join(reasons)}


def munger_agent(fund):
    """Charlie Munger - quality at fair price, ROIC high."""
    if not fund or not fund.get('roa'):
        return {'agent': 'munger', 'signal': 'neutral', 'confidence': 30, 'reasoning': 'Sin datos suficientes'}

    roa = fund['roa']
    pm = fund.get('profit_margin', 0)
    pe = fund.get('pe_ratio', 100)
    score = 0
    reasons = []

    if roa > 0.15:
        score += 30
        reasons.append(f'ROA {roa*100:.1f}% > 15% (calidad alta)')
    elif roa > 0.10:
        score += 20
    else:
        reasons.append(f'ROA {roa*100:.1f}% debil')

    if pm > 0.20:
        score += 25
    elif pm > 0.10:
        score += 10

    if pe > 0 and pe < 30:
        score += 20
    elif pe > 50:
        score -= 15

    if fund.get('roe', 0) > 0.15:
        score += 15

    signal = 'bullish' if score >= 70 else ('bearish' if score <= 30 else 'neutral')
    return {'agent': 'munger', 'signal': signal, 'confidence': min(100, max(0, score)), 'reasoning': '; '.join(reasons)}


def lynch_agent(fund):
    """Peter Lynch - growth at reasonable price (PEG)."""
    if not fund or 'peg_ratio' not in fund:
        return {'agent': 'lynch', 'signal': 'neutral', 'confidence': 30, 'reasoning': 'Sin datos suficientes'}

    peg = fund.get('peg_ratio', 0)
    rg = fund.get('revenue_growth', 0)
    eps = fund.get('eps', 0)
    score = 0
    reasons = []

    if peg > 0 and peg < 0.5:
        score += 35
        reasons.append(f'PEG {peg:.2f} < 0.5 (deep value growth!)')
    elif peg > 0 and peg < 1.0:
        score += 30
        reasons.append(f'PEG {peg:.2f} < 1.0')
    elif peg > 0 and peg < 1.5:
        score += 15
    elif peg > 1.5:
        score -= 10
        reasons.append(f'PEG {peg:.2f} caro')

    if rg > 0.20:
        score += 25
    elif rg > 0.10:
        score += 15

    if eps > 0:
        score += 15

    if fund.get('dividend_yield', 0) > 0.02:
        score += 10

    signal = 'bullish' if score >= 70 else ('bearish' if score <= 30 else 'neutral')
    return {'agent': 'lynch', 'signal': signal, 'confidence': min(100, max(0, score)), 'reasoning': '; '.join(reasons)}


def drunkenmiller_agent(fund, mom_12m=None):
    """Stanley Druckenmiller - macro trends + momentum + fundamentals."""
    if not fund:
        return {'agent': 'drunkenmiller', 'signal': 'neutral', 'confidence': 30, 'reasoning': 'Sin datos suficientes'}

    rg = fund.get('revenue_growth', 0)
    pe = fund.get('pe_ratio', 100)
    sector = fund.get('sector', '')
    score = 0
    reasons = []

    if mom_12m is not None:
        if mom_12m > 30:
            score += 30
            reasons.append(f'Momentum 12m {mom_12m:.1f}% fuerte')
        elif mom_12m > 10:
            score += 20
        elif mom_12m < -10:
            score -= 15
            reasons.append(f'Momentum 12m {mom_12m:.1f}% negativo')

    if rg > 0.20:
        score += 25
        reasons.append(f'Revenue growth {rg*100:.1f}% (macro tailwind)')
    elif rg > 0.10:
        score += 15

    if sector in ('TECHNOLOGY', 'ENERGY', 'FINANCE'):
        score += 15
        reasons.append(f'Sector {sector} en tendencia')

    if pe > 0 and pe < 30:
        score += 15

    signal = 'bullish' if score >= 65 else ('bearish' if score <= 30 else 'neutral')
    return {'agent': 'drunkenmiller', 'signal': signal, 'confidence': min(100, max(0, score)), 'reasoning': '; '.join(reasons)}


def aggregate_signals(agents_results, weights=None):
    """Combina las senales de los 5 agentes en un score final."""
    if weights is None:
        weights = {'graham': 2.0, 'buffett': 1.0, 'munger': 1.0, 'lynch': 1.0, 'drunkenmiller': 1.0}

    signal_map = {'bullish': 1, 'neutral': 0, 'bearish': -1}
    total_weight = sum(weights.values())
    weighted_score = 0
    total_confidence = 0
    detailed = []

    for r in agents_results:
        agent = r['agent']
        w = weights.get(agent, 1.0)
        s = signal_map.get(r['signal'], 0)
        c = r.get('confidence', 50) / 100.0
        weighted_score += s * c * w
        total_confidence += c * w
        detailed.append({
            'agent': agent,
            'signal': r['signal'],
            'confidence': r.get('confidence', 0),
            'reasoning': r.get('reasoning', '')[:100],
            'weight': w,
        })

    final_score = weighted_score / total_weight if total_weight > 0 else 0

    if final_score > 0.3:
        final_signal = 'bullish'
    elif final_score < -0.2:
        final_signal = 'bearish'
    else:
        final_signal = 'neutral'

    return {
        'final_signal': final_signal,
        'final_score': round(final_score, 3),
        'final_confidence': round(total_confidence / total_weight * 100, 1),
        'agents': detailed,
    }


def analyze_stock(symbol, mom_12m=None):
    """Analiza un stock con los 5 agentes y devuelve score agregado."""
    fund = get_fundamentals(symbol)
    if not fund:
        return {'symbol': symbol, 'error': 'Sin fundamentales', 'final_signal': 'neutral'}

    agents = [
        graham_agent(fund),
        buffett_agent(fund),
        munger_agent(fund),
        lynch_agent(fund),
        drunkenmiller_agent(fund, mom_12m),
    ]
    agg = aggregate_signals(agents)
    return {
        'symbol': symbol,
        'fundamentals': fund,
        'agents': agents,
        **agg,
    }