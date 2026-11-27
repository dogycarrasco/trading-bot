"""
Sentiment layer con VADER (liviano, para CI en GitHub Actions).

Mantiene la interfaz del finbert_sentiment original:
  - FinBERTSentiment class
  - fetch_rss_headlines(max_per_feed)
  - filter_headlines_by_asset(headlines, symbol)

VADER no requiere descarga de modelo, corre en <1s, y es suficiente
para un bot con otras 4 capas + circuit breaker.
"""
import re
import time
from typing import Dict, List

import requests

# VADER: liviano (~100KB), sin download de modelo
try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    _ANALYZER = SentimentIntensityAnalyzer()
    HAS_VADER = True
except ImportError:
    _ANALYZER = None
    HAS_VADER = False

# RSS feeds financieros (gratuitos, sin API key)
RSS_FEEDS = [
    "https://finance.yahoo.com/news/rssindex",
    "https://www.cnbc.com/id/100003114/device/rss/rss.html",
    "https://feeds.content.dowjones.io/public/rss/mw_topstories",
]

# symbol → keywords para filtrar headlines relevantes
ASSET_KEYWORDS = {
    "SPY": ["s&p 500", "sp500", "spy etf", "stock market", "stocks", "equities"],
    "QQQ": ["nasdaq", "qqq", "tech stocks", "technology sector"],
    "XLK": ["technology", "semiconductor", "software", "chip"],
    "XLF": ["banks", "financials", "banking sector", "jpmorgan", "goldman"],
    "XLE": ["oil", "energy sector", "crude", "petroleum", "opec"],
    "XLV": ["healthcare", "pharma", "biotech", "medical", "drug"],
    "XLY": ["consumer discretionary", "retail", "amazon", "tesla"],
    "XLP": ["consumer staples", "grocery", "walmart", "procter"],
    "XLI": ["industrials", "manufacturing", "boeing", "caterpillar"],
    "XLU": ["utilities", "utility sector", "electric power"],
    "GLD": ["gold", "precious metals", "bullion"],
    "USO": ["oil", "crude", "wti", "brent"],
    "BTCUSD": ["bitcoin", "btc", "crypto"],
    "ETHUSD": ["ethereum", "eth", "crypto"],
    "VIX": ["volatility", "vix", "fear index"],
}

# Fallback si VADER no disponible: keyword counting
BULLISH_WORDS = {
    "surge", "rally", "gain", "rise", "soar", "jump", "beat", "record",
    "upgrade", "buy", "bullish", "strong", "growth", "profit", "outperform",
    "recover", "rebound", "optimism", "boost", "positive", "higher", "up",
}
BEARISH_WORDS = {
    "fall", "drop", "plunge", "decline", "crash", "sell", "bearish", "weak",
    "loss", "downgrade", "miss", "warning", "fear", "panic", "risk",
    "recession", "crisis", "tumble", "slump", "negative", "lower", "down",
}


class FinBERTSentiment:
    """
    Analizador de sentiment (VADER + fallback keywords).
    Misma interfaz que finBERT original.
    """

    def __init__(self):
        self.analyzer = _ANALYZER
        self.available = HAS_VADER

    def analyze_sentiment(self, text: str) -> float:
        """Devuelve compound score [-1, 1]. >0.05 positivo, <-0.05 negativo."""
        if self.analyzer is not None:
            scores = self.analyzer.polarity_scores(text)
            return scores["compound"]
        return self._keyword_score(text)

    def _keyword_score(self, text: str) -> float:
        """Fallback: cuenta palabras bullish/bearish."""
        words = set(re.findall(r"\w+", text.lower()))
        bull = len(words & BULLISH_WORDS)
        bear = len(words & BEARISH_WORDS)
        total = bull + bear
        return (bull - bear) / total if total > 0 else 0.0

    def get_asset_sentiment(self, headlines: List[str], symbol: str) -> Dict:
        """Agrega sentiment de headlines relevantes a un symbol."""
        relevant = filter_headlines_by_asset(headlines, symbol)
        if not relevant:
            return {"vote": 0, "score": 0.0, "n_headlines": 0,
                    "reason": f"no headlines for {symbol}"}

        scores = [self.analyze_sentiment(h) for h in relevant]
        avg = sum(scores) / len(scores)

        if avg > 0.15:
            vote = 1
        elif avg < -0.15:
            vote = -1
        else:
            vote = 0

        return {"vote": vote, "score": round(avg, 3),
                "n_headlines": len(relevant),
                "reason": f"avg compound {avg:+.3f} x {len(relevant)} headlines"}


def fetch_rss_headlines(max_per_feed: int = 10) -> List[str]:
    """Descarga headlines de RSS feeds financieros. Sin API key."""
    headlines = []
    for feed_url in RSS_FEEDS:
        try:
            resp = requests.get(feed_url, timeout=10,
                                headers={"User-Agent": "Mozilla/5.0 TradingBot/1.0"})
            if resp.status_code != 200:
                continue
            titles = re.findall(
                r"<title>(?:<!\[CDATA\[)?(.*?)(?:\]\]>)?</title>",
                resp.text, re.DOTALL)
            # titles[0] es el titulo del feed, no de un articulo
            for title in titles[1:max_per_feed + 1]:
                clean = title.strip()
                if clean and len(clean) > 10:
                    headlines.append(clean)
        except Exception:
            continue
        time.sleep(0.5)
    return headlines


def filter_headlines_by_asset(headlines: List[str], symbol: str) -> List[str]:
    """Filtra headlines relevantes para un symbol."""
    keywords = ASSET_KEYWORDS.get(symbol, [symbol.lower()])
    keywords_lower = [k.lower() for k in keywords]
    return [h for h in headlines
            if any(kw in h.lower() for kw in keywords_lower)]
