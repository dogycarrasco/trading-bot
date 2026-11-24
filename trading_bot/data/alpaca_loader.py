"""Alpaca data adapter via REST API con Basic Auth."""
import os
import requests
import pandas as pd

API_KEY = os.environ.get("ALPACA_API_KEY", "")
SECRET_KEY = os.environ.get("ALPACA_SECRET_KEY", "")
DATA_URL = "https://data.alpaca.markets/v2/stocks"
ACCOUNT_URL = "https://paper-api.alpaca.markets/v2/account"

AUTH = (API_KEY, SECRET_KEY) if API_KEY and SECRET_KEY else None

_keys_ok = None  # cache: None=no verificado, True/False=resultado

def verify_keys():
    global _keys_ok
    if _keys_ok is not None:
        return _keys_ok
    if not AUTH:
        print(f"Alpaca: API_KEY={'si' if API_KEY else 'no'}({len(API_KEY)}) SECRET={'si' if SECRET_KEY else 'no'}({len(SECRET_KEY)})")
        _keys_ok = False
        return False
    print(f"Alpaca debug: key='{API_KEY[:5]}...' secret='{SECRET_KEY[:5]}...'")
    try:
        r = requests.get(ACCOUNT_URL, auth=AUTH, timeout=10)
        if r.status_code == 200:
            acct = r.json()
            print(f"Alpaca keys OK: paper ${float(acct.get('cash',0)):,.0f}")
            _keys_ok = True
        else:
            print(f"Alpaca keys FAIL: HTTP {r.status_code}")
            _keys_ok = False
    except Exception as e:
        print(f"Alpaca keys ERROR: {e}")
        _keys_ok = False
    return _keys_ok


def fetch_alpaca_bars(symbol, limit=500):
    if not verify_keys():
        return pd.DataFrame()
    try:
        url = f"{DATA_URL}/{symbol}/bars"
        params = {"timeframe": "1Day", "limit": limit, "feed": "iex", "sort": "asc"}
        response = requests.get(url, auth=AUTH, params=params, timeout=15)
        if response.status_code != 200:
            return pd.DataFrame()
        bars = response.json().get("bars", [])
        if not bars:
            return pd.DataFrame()
        df = pd.DataFrame(bars)
        df["date"] = pd.to_datetime(df["t"])
        df = df.set_index("date").rename(columns={"o":"open","h":"high","l":"low","c":"close","v":"volume"})
        df = df[["open","high","low","close","volume"]].dropna(subset=["close"])
        print(f"Alpaca {symbol}: {len(df)} bars OK")
        return df
    except Exception:
        return pd.DataFrame()


def load_via_alpaca(symbol, exchange=""):
    alpaca_map = {"SPY":"SPY","QQQ":"QQQ","XLK":"XLK","XLF":"XLF","XLE":"XLE","XLV":"XLV",
        "XLY":"XLY","XLP":"XLP","XLI":"XLI","XLU":"XLU","GLD":"GLD","USO":"USO"}
    sym = alpaca_map.get(symbol)
    if sym is None or not API_KEY:
        return pd.DataFrame()
    return fetch_alpaca_bars(sym)
