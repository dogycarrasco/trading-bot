"""
Generacion de labels para el modelo ML.

Label: direccion del retorno a N barras (1=up, 0=flat, -1=down).
Threshold adaptativo: 0.5 * std(returns_historicos) para que cada activo tenga
su propia escala.
"""
import numpy as np
import pandas as pd


def adaptive_threshold(close, horizon, multiplier=0.5):
    """Threshold basado en volatilidad historica. Sin lookahead."""
    fwd_ret = close.shift(-horizon) / close - 1
    std = fwd_ret.std()
    if pd.isna(std) or std <= 0:
        return 0.01
    return float(std * multiplier)


def make_label(close, horizon=5, threshold=None):
    """
    Label basado en retorno forward a `horizon` barras.
    +1 si retorno > threshold
    -1 si retorno < -threshold
     0 si |retorno| <= threshold
    Si threshold es None, lo calcula adaptativo (0.5 * std del forward return).
    """
    if threshold is None:
        threshold = adaptive_threshold(close, horizon, multiplier=0.5)
    fwd_ret = close.shift(-horizon) / close - 1
    label = np.where(fwd_ret > threshold, 1, np.where(fwd_ret < -threshold, -1, 0))
    return pd.Series(label, index=close.index)


def label_distribution(label):
    counts = label.value_counts().sort_index()
    total = len(label)
    mapping = {-1: "DOWN", 0: "FLAT", 1: "UP"}
    return {mapping.get(k, str(k)): f"{v} ({v/total*100:.1f}%)" for k, v in counts.items()}