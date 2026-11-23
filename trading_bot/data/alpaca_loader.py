"""Alpaca data adapter — reemplaza yfinance para datos de precios.
Usa Alpaca Market Data API para barras diarias en tiempo real.
Fallback a yfinance si Alpaca no disponible."""
import os
import pandas as pd
from datetime import datetime, timedelta

try:
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame
    ALPACA_AVAILABLE = True
except ImportError:
    ALPACA_AVAILABLE = False

API_KEY = os.environ.get("ALPACA_API_KEY", "")
SECRET_KEY = os.environ.get("ALPACA_SECRET_KEY", "")

_client = None

def _get_client():
    global _client
    if _client is None and API_KEY and SECRET_KEY:
        _client = StockHistoricalDataClient(API_KEY, SECRET_KEY)
    return _client


def fetch_alpaca_bars(symbol, days=500):
    """Descarga barras diarias de Alpaca. Devuelve DataFrame con close/high/low/open/volume."""
    client = _get_client()
    if client is None:
        return pd.DataFrame()
    try:
        end = datetime.now()
        start = end - timedelta(days=days)
        request = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=TimeFrame.Day,
            start=start,
            end=end,
        )
        bars = client.get_stock_bars(request)
        if bars is None or len(bars) == 0:
            return pd.DataFrame()
        df = bars.df if hasattr(bars, "df") else pd.DataFrame(bars)
        if isinstance(df.index, pd.MultiIndex):
            df = df.droplevel(0)
        # Normalizar columnas
        df.columns = [c.lower() for c in df.columns]
        df = df.dropna(subset=["close"])
        return df
    except Exception as e:
        print(f"Alpaca {symbol}: ERROR {type(e).__name__}: {str(e)[:60]}")
        return pd.DataFrame()


def load_via_alpaca(symbol, exchange=""):
    """Intenta cargar datos de Alpaca primero, fallback a yfinance."""
    # Map: bot symbols -> Alpaca symbols
    alpaca_map = {
        "SPY": "SPY", "QQQ": "QQQ", "XLK": "XLK", "XLF": "XLF",
        "XLE": "XLE", "XLV": "XLV", "XLY": "XLY", "XLP": "XLP",
        "XLI": "XLI", "XLU": "XLU", "GLD": "GLD", "USO": "USO",
        "BTCUSD": None, "ETHUSD": None, "SOLUSD": None,
        "BNBUSD": None, "XRPUSD": None, "ADAUSD": None,
        "VIX": None,  # VIX no disponible en Alpaca
    }
    alpaca_sym = alpaca_map.get(symbol)
    if alpaca_sym is None or not ALPACA_AVAILABLE or not API_KEY:
        return pd.DataFrame()  # crypto/VIX: usar yfinance fallback
    
    return fetch_alpaca_bars(alpaca_sym)
