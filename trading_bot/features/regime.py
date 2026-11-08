"""
Detector de regimen de mercado.

Clasifica cada barra en uno de 4 regimenes:
  - TREND_UP:    tendencia alcista
  - TREND_DOWN:  tendencia bajista
  - RANGE:       lateral
  - HIGH_VOL:    alta volatilidad
"""
import numpy as np
import pandas as pd


def adaptive_vol_threshold(atr_pct, percentile=85.0):
    """
    Threshold de volatilidad adaptativo: percentil alto del ATR% historico.
    Sin lookahead: usa todo el historico disponible (percentil estable).
    """
    if atr_pct.empty:
        return 0.025
    return float(np.nanpercentile(atr_pct.dropna(), percentile))


def detect_regime(df, adx_thresh=25.0, vol_thresh=None):
    """
    Recibe un df con features ya calculados (adx_14, atr_pct, ema_*, close).
    Devuelve Series de strings con el regimen por barra.

    Si vol_thresh es None, usa el percentil 85 del ATR% del propio simbolo.
    """
    atr_pct = df["atr_pct"]
    if vol_thresh is None:
        vol_thresh = adaptive_vol_threshold(atr_pct, percentile=85.0)

    adx = df["adx_14"]
    close = df["close"]
    ema50 = df["ema_50"]
    ema9 = df["ema_9"]
    ema21 = df["ema_21"]

    reg = pd.Series("RANGE", index=df.index, dtype="object")

    trending = adx >= adx_thresh
    up = trending & (close > ema50) & (ema9 > ema21)
    down = trending & (close < ema50) & (ema9 < ema21)
    high_vol = atr_pct >= vol_thresh

    reg[up] = "TREND_UP"
    reg[down] = "TREND_DOWN"
    reg[high_vol & ~up & ~down] = "HIGH_VOL"

    return reg


def regime_to_int(reg):
    """Mapping a enteros para features del ML."""
    m = {"RANGE": 0, "TREND_UP": 1, "TREND_DOWN": 2, "HIGH_VOL": 3}
    return reg.map(m).fillna(0).astype(int)


def regime_summary(reg):
    """Distribucion de regimenes, util para reporte."""
    counts = reg.value_counts()
    total = len(reg)
    return {k: f"{v}/{total} ({v/total*100:.1f}%)" for k, v in counts.items()}