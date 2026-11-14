"""
v38 predictive models: GMM crisis + PCA factor + EWMA vol + RSI filter.
"""
import os, sys, warnings, numpy as np, pandas as pd
from collections import deque

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

try:
    from sklearn.mixture import GaussianMixture
    from sklearn.decomposition import PCA
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

def rsi(close, period=14):
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1/period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1/period, adjust=False).mean()
    return (100 - 100 / (1 + gain / loss.replace(0, np.nan))).fillna(50)

def ema(s, p): return s.ewm(span=p, adjust=False).mean()
def vwap_r(c, v, w=20): return (c * v).rolling(w).sum() / v.rolling(w).sum()

def rsi_ema_vwap_filter(df, date):
    """RSI+EMA+VWAP: NUNCA bloquea, solo ajusta size."""
    ts = pd.Timestamp(date)
    sub = df.loc[:ts]
    if len(sub) < 105: return 1.0
    c = sub["close"]; v = sub["volume"]
    r4 = float(rsi(c, 4).iloc[-1])
    e100 = float(ema(c, 100).iloc[-1])
    cv = float(c.iloc[-1])
    vw = float(vwap_r(c, v).iloc[-1]) if not pd.isna(vwap_r(c, v).iloc[-1]) else cv
    aligned = sum([r4 < 30, cv > e100, cv < vw])
    if aligned == 3: return 1.0
    elif aligned == 2: return 0.8
    elif aligned == 1: return 0.6
    return 0.4

def ewma_vol_at_date(df, date, alpha=0.06):
    """EWMA volatility anualizada hasta una fecha."""
    if df is None or df.empty: return 0.20
    ts = pd.Timestamp(date)
    sub = df.loc[:ts]
    if len(sub) < 30: return 0.20
    r = sub["close"].pct_change().dropna()
    vol = np.sqrt(r.ewm(alpha=alpha).var().iloc[-1]) * np.sqrt(252)
    return float(vol) if not pd.isna(vol) else 0.20

class GMMCrisis:
    def __init__(self):
        self.model = None; self.crisis_label = None; self.last_train = 0
    def _feats(self, spy_df, date):
        ts = pd.Timestamp(date); sub = spy_df.loc[:ts]
        if len(sub) < 60: return None
        c = sub["close"]; r = c.pct_change().dropna()
        return pd.DataFrame({"ret": r, "vs": r.rolling(5).std(), "vl": r.rolling(20).std(), "m": r.rolling(10).mean()}).dropna()
    def train(self, spy_df, date):
        if not HAS_SKLEARN: return False
        f = self._feats(spy_df, date)
        if f is None or len(f) < 252: return False
        try:
            self.model = GaussianMixture(n_components=3, covariance_type="full", random_state=42, n_init=3)
            self.model.fit(f.values)
            self.crisis_label = int(np.argmax([m[2] for m in self.model.means_]))
            self.last_train = len(f); return True
        except: return False
    def crisis_prob(self, spy_df, date):
        if self.model is None: return 0.0
        f = self._feats(spy_df, date)
        if f is None or len(f) == 0: return 0.0
        try:
            p = self.model.predict_proba(f.iloc[-1:].values)[0]
            return float(p[self.crisis_label]) if self.crisis_label is not None else 0.0
        except: return 0.0
    def should_retrain(self, spy_df, date):
        f = self._feats(spy_df, date)
        return f is not None and len(f) - self.last_train >= 60

class PCAFactor:
    def __init__(self):
        self.model = None; self.mean = 0; self.std = 1; self.last_train = 0
    def train(self, asset_dfs, date):
        if not HAS_SKLEARN: return False
        ts = pd.Timestamp(date); rd = {}
        for sym, df in asset_dfs.items():
            sub = df.loc[:ts]
            if len(sub) < 252: continue
            r = sub["close"].pct_change().dropna()
            if len(r) >= 252: rd[sym] = r
        if len(rd) < 3: return False
        dr = pd.DataFrame(rd).dropna()
        if len(dr) < 252: return False
        try:
            self.model = PCA(n_components=1, random_state=42)
            self.model.fit(dr.values)
            scores = self.model.transform(dr.values)[:, 0]
            self.mean = float(np.mean(scores[-252:])); self.std = float(np.std(scores[-252:]))
            self.last_train = len(dr); return True
        except: return False
    def zscore(self, asset_dfs, date):
        if self.model is None: return 0.0
        ts = pd.Timestamp(date); rd = {}
        for sym, df in asset_dfs.items():
            sub = df.loc[:ts]
            if len(sub) < 2: continue
            r = sub["close"].pct_change().iloc[-1]
            if not pd.isna(r): rd[sym] = r
        if len(rd) < 3: return 0.0
        try:
            f = self.model.transform(np.array([list(rd.values())]))[0, 0]
            return float((f - self.mean) / self.std) if self.std > 0 else 0.0
        except: return 0.0
