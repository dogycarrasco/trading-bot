"""Alpaca data adapter via REST API — sin librería alpaca-py.
Usa requests + HTTP directo a Alpaca Market Data API v2 (feed IEX gratuito).
Fallback a yfinance si Alpaca no disponible."""
import os
import requests
import pandas as pd

API_KEY = os.environ.get("ALPACA_API_KEY", "")
SECRET_KEY = os.environ.get("ALPACA_SECRET_KEY", "")
DATA_URL = "https://data.alpaca.markets/v2/stocks"

HEADERS = {
    "APCA-API-KEY-ID": API_KEY,
    "APCA-API-SECRET-KEY": SECRET_KEY,
}


_keys_verified = False

def verify_keys():
    """Verifica que las API keys funcionan con el endpoint de cuenta (paper)."""
    global _keys_verified
    if _keys_verified:
        return True
    if not API_KEY or not SECRET_KEY:
        print("Alpaca: sin API keys en env")
        return False
    try:
        r = requests.get("https://paper-api.alpaca.markets/v2/account", headers=HEADERS, timeout=10)
        if r.status_code == 200:
            acct = r.json()
            print(f"Alpaca keys OK: cuenta paper ${float(acct.get('cash',0)):,.0f}")
            _keys_verified = True
            return True
        else:
            print(f"Alpaca keys FAIL: account HTTP {r.status_code}")
            return False
    except Exception as e:
        print(f"Alpaca keys ERROR: {e}")
        return False


def fetch_alpaca_bars(symbol, limit=500):
    """Descarga barras diarias de Alpaca via REST API (feed IEX gratuito)."""
    if not API_KEY or not SECRET_KEY:
        print(f"Alpaca {symbol}: sin API keys")
        return pd.DataFrame()
    if not verify_keys():
        return pd.DataFrame()
    try:
        url = f"{DATA_URL}/{symbol}/bars"
        params = {
            "timeframe": "1Day",
            "limit": limit,
            "feed": "iex",
            "sort": "asc",
        }
        response = requests.get(url, headers=HEADERS, params=params, timeout=15)

        if response.status_code != 200:
            print(f"Alpaca {symbol}: HTTP {response.status_code} - {response.text[:100]}")
            return pd.DataFrame()

        data = response.json()
        bars = data.get("bars", [])

        if not bars:
            print(f"Alpaca {symbol}: sin barras en respuesta")
            return pd.DataFrame()

        # Convertir a DataFrame
        df = pd.DataFrame(bars)
        df["date"] = pd.to_datetime(df["t"])
        df = df.set_index("date")
        df = df.rename(columns={"o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"})
        df = df[["open", "high", "low", "close", "volume"]]
        df = df.dropna(subset=["close"])

        # Paginación (next_page_token)
        next_token = data.get("next_page_token")
        while next_token and len(df) < 2000:
            params["page_token"] = next_token
            resp2 = requests.get(url, headers=HEADERS, params=params, timeout=15)
            if resp2.status_code != 200:
                break
            more = resp2.json().get("bars", [])
            if not more:
                break
            more_df = pd.DataFrame(more)
            more_df["date"] = pd.to_datetime(more_df["t"])
            more_df = more_df.set_index("date")
            more_df = more_df.rename(columns={"o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"})
            more_df = more_df[["open", "high", "low", "close", "volume"]]
            more_df = more_df.dropna(subset=["close"])
            df = pd.concat([df, more_df])
            next_token = resp2.json().get("next_page_token")

        print(f"Alpaca {symbol}: {len(df)} barras OK (feed iex)")
        return df

    except Exception as e:
        print(f"Alpaca {symbol}: {type(e).__name__}: {str(e)[:120]}")
        return pd.DataFrame()


def load_via_alpaca(symbol, exchange=""):
    """Intenta cargar datos de Alpaca primero, fallback a yfinance."""
    alpaca_map = {
        "SPY": "SPY", "QQQ": "QQQ", "XLK": "XLK", "XLF": "XLF",
        "XLE": "XLE", "XLV": "XLV", "XLY": "XLY", "XLP": "XLP",
        "XLI": "XLI", "XLU": "XLU", "GLD": "GLD", "USO": "USO",
        "BTCUSD": None, "ETHUSD": None, "SOLUSD": None,
        "BNBUSD": None, "XRPUSD": None, "ADAUSD": None,
        "VIX": None,
    }
    alpaca_sym = alpaca_map.get(symbol)
    if alpaca_sym is None or not API_KEY:
        return pd.DataFrame()

    return fetch_alpaca_bars(alpaca_sym)
