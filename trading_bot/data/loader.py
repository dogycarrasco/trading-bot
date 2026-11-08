"""
Carga de datos historical para backtest y paper trading.

Binance: API publico, no requiere API key para datos.
yfinance: proxy para acciones hasta activar Alpaca paper.
TradingView (tvDatafeed): fuente principal para ETFs, futuros, cripto, VIX.
"""
import time
import requests
import pandas as pd
import yfinance as yf


def _binance_klines(symbol, interval, start_ms, end_ms, limit=1000):
    """Llama al endpoint publico de Binance. No requiere auth."""
    url = "https://api.binance.com/api/v3/klines"
    all_rows = []
    cur = start_ms
    while cur < end_ms:
        params = {"symbol": symbol, "interval": interval, "startTime": cur, "endTime": end_ms, "limit": limit}
        r = requests.get(url, params=params, timeout=30)
        r.raise_for_status()
        rows = r.json()
        if not rows:
            break
        all_rows.extend(rows)
        last_open = rows[-1][0]
        if last_open <= cur:
            break
        cur = last_open + 1
        time.sleep(0.15)
    return all_rows


def load_binance(symbol, timeframe="1d", years=2):
    """Descarga historico de Binance spot."""
    interval_map = {"1d": "1d", "1h": "1h", "4h": "4h", "15m": "15m", "5m": "5m"}
    if timeframe not in interval_map:
        raise ValueError(f"Timeframe {timeframe} no soportado para Binance")
    interval = interval_map[timeframe]

    end_ms = int(time.time() * 1000)
    start_ms = end_ms - years * 365 * 24 * 3600 * 1000

    rows = _binance_klines(symbol, interval, start_ms, end_ms)
    if not rows:
        raise RuntimeError(f"Sin datos para {symbol} en Binance")

    df = pd.DataFrame(rows, columns=[
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "trades", "taker_buy_base",
        "taker_buy_quote", "ignore"
    ])
    df["datetime"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df = df[["datetime", "open", "high", "low", "close", "volume"]].copy()
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = df[c].astype(float)
    df = df.set_index("datetime").sort_index()
    df = df[~df.index.duplicated(keep="last")]
    return df


def load_stock(symbol, timeframe="1d", years=2):
    """
    Descarga historico de acciones via yfinance.
    yfinance no soporta 4h nativo; si se pide 4h, bajamos 1h y resampleamos.
    """
    if timeframe == "1d":
        interval = "1d"
        period = f"{years}y"
    elif timeframe in ("4h", "4H"):
        interval = "60m"
        period = "730d"
    elif timeframe in ("1h", "60m"):
        interval = "60m"
        period = "730d"
    elif timeframe == "15m":
        interval = "15m"
        period = "60d"
    else:
        raise ValueError(f"Timeframe {timeframe} no soportado para stocks")

    df = yf.download(symbol, period=period, interval=interval, progress=False, auto_adjust=True)
    if df.empty:
        raise RuntimeError(f"Sin datos para {symbol} en yfinance")

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    rename = {"Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"}
    df = df.rename(columns=rename)[["open", "high", "low", "close", "volume"]].copy()
    df.index = df.index.tz_localize("UTC") if df.index.tz is None else df.index.tz_convert("UTC")
    df.index.name = "datetime"
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = df[c].astype(float)
    df = df[~df.index.duplicated(keep="last")].sort_index()

    if timeframe in ("4h", "4H") and interval == "60m":
        agg = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
        df = df.resample("4h").agg(agg).dropna()

    return df


def load_series(symbol, broker, timeframe="1d", years=2):
    """Dispatcher unico: devuelve OHLCV para cualquier symbol segun broker."""
    if broker in ("binance", "binance_paper"):
        return load_binance(symbol, timeframe, years)
    if broker in ("alpaca", "alpaca_proxy", "ibkr", "ibkr_paper"):
        return load_stock(symbol, timeframe, years)
    raise ValueError(f"Broker {broker} no soportado")


if __name__ == "__main__":
    print("Descargando BTCUSDT (Binance)...")
    btc = load_binance("BTCUSDT", "1d", 2)
    print(f"  BTCUSDT: {len(btc)} barras, last close={btc['close'].iloc[-1]:.2f}")
    print("Descargando SPY (yfinance)...")
    spy = load_stock("SPY", "1d", 2)
    print(f"  SPY: {len(spy)} barras, last close={spy['close'].iloc[-1]:.2f}")