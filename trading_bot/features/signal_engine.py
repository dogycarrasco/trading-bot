"""
Signal Engine v1 - Multi-Signal Voting Framework.

5 capas votan (+1, 0, -1). Solo se compra si >= 3 capas dicen +1.
"""
import os
import sys
import json
import warnings
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from features.options_hedge import compute_market_score, score_to_regime
from backtest.run_v8_hedge_fund import (
    UNIVERSE_V8, load_tv_data, load_macro_features,
)


def _ema(series, period):
    return series.ewm(span=period, adjust=False).mean()


def _rsi(close, period=14):
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1/period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1/period, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(50)


def _safe_loc(df, date):
    if df is None or df.empty:
        return None
    try:
        sub = df.loc[:date]
        return sub if not sub.empty else None
    except Exception:
        return None


class RegimeSignal:
    name = "regime"

    def __init__(self):
        self.features_df = None
        self.spy_df = None

    def _load(self):
        if self.features_df is None:
            self.features_df = load_macro_features()
        if self.spy_df is None:
            self.spy_df = load_tv_data("SPY", "")

    def evaluate(self, date):
        self._load()
        if self.features_df is None or self.features_df.empty:
            return {"vote": 0, "confidence": 0, "reason": "no_macro_data"}
        feat = _safe_loc(self.features_df, pd.Timestamp(date))
        if feat is None:
            return {"vote": 0, "confidence": 0, "reason": "no_macro_at_date"}
        last = feat.iloc[-1]
        vix = float(last.get("vix", 0)) if pd.notna(last.get("vix")) else None
        fed_funds = float(last.get("fed_funds", 0)) if pd.notna(last.get("fed_funds")) else None
        yield_spread = float(last.get("yield_spread", 0)) if pd.notna(last.get("yield_spread")) else None
        fear_greed = float(last.get("fear_greed", 50)) if pd.notna(last.get("fear_greed")) else None
        spy_sub = _safe_loc(self.spy_df, pd.Timestamp(date))
        if spy_sub is None or len(spy_sub) < 200:
            return {"vote": 0, "confidence": 0, "reason": "no_spy_history"}
        spy_close = spy_sub["close"]
        ema200 = float(_ema(spy_close, 200).iloc[-1])
        spy_price = float(spy_close.iloc[-1])
        spy_above_ema200 = spy_price > ema200
        mom_20d = (spy_price / float(spy_close.iloc[-21]) - 1) * 100 if len(spy_close) > 21 else 0
        mom_50d = (spy_price / float(spy_close.iloc[-51]) - 1) * 100 if len(spy_close) > 51 else 0
        yield_inverted = (yield_spread is not None and yield_spread < 0)
        score = compute_market_score(vix, fear_greed, spy_above_ema200, yield_inverted, fed_funds, mom_20d, mom_50d)
        regime = score_to_regime(score)
        if score >= 20:
            vote, conf = 1, min(1.0, score / 100)
        elif score <= -20:
            vote, conf = -1, min(1.0, abs(score) / 100)
        else:
            vote, conf = 0, 0.3
        return {"vote": vote, "confidence": round(conf, 2), "reason": f"score={score:+.0f} regime={regime}", "score": score, "regime": regime}


class SentimentSignal:
    name = "sentiment"

    def __init__(self):
        self.fg_df = None

    def _load(self):
        if self.fg_df is None:
            path = os.path.join(ROOT, "data", "external", "fear_greed.csv")
            if os.path.exists(path):
                try:
                    df = pd.read_csv(path)
                    if "date" in df.columns:
                        df["date"] = pd.to_datetime(df["date"])
                        df = df.set_index("date").sort_index()
                        self.fg_df = df
                except Exception:
                    self.fg_df = pd.DataFrame()

    def evaluate(self, date):
        self._load()
        if self.fg_df is None or self.fg_df.empty:
            return {"vote": 0, "confidence": 0, "reason": "no_fg_data"}
        sub = _safe_loc(self.fg_df, pd.Timestamp(date))
        if sub is None or sub.empty:
            return {"vote": 0, "confidence": 0, "reason": "no_fg_at_date"}
        col = None
        for c in ["value", "fear_greed", "fg", "score"]:
            if c in sub.columns:
                col = c
                break
        if col is None:
            return {"vote": 0, "confidence": 0, "reason": "no_fg_col"}
        fg = float(sub[col].iloc[-1])
        if fg >= 60:
            vote, conf = 1, (fg - 50) / 50
        elif fg <= 25:
            vote, conf = -1, (50 - fg) / 50
        else:
            vote, conf = 0, 0.3
        return {"vote": vote, "confidence": round(conf, 2), "reason": f"fg={fg:.0f}", "fg": fg}


class OnchainSignal:
    name = "onchain"

    def __init__(self):
        self.btc_df = None

    def _load(self):
        if self.btc_df is None:
            for sym in ["BTCUSD", "BTCUSDT", "BTC"]:
                df = load_tv_data(sym, "")
                if not df.empty:
                    self.btc_df = df
                    return

    def evaluate(self, date, is_crypto=False):
        self._load()
        if self.btc_df is None or self.btc_df.empty:
            return {"vote": 0, "confidence": 0, "reason": "no_btc_data"}
        sub = _safe_loc(self.btc_df, pd.Timestamp(date))
        if sub is None or len(sub) < 200:
            return {"vote": 0, "confidence": 0, "reason": "no_btc_history"}
        close = sub["close"]
        ema200 = float(_ema(close, 200).iloc[-1])
        price = float(close.iloc[-1])
        ratio = price / ema200 if ema200 > 0 else 1
        if not is_crypto:
            return {"vote": 0, "confidence": 0.2, "reason": f"btc r={ratio:.2f}", "btc_ratio": ratio}
        if ratio > 2.5:
            return {"vote": -1, "confidence": 0.7, "reason": f"bubble r={ratio:.2f}", "btc_ratio": ratio}
        elif ratio < 0.8:
            return {"vote": -1, "confidence": 0.4, "reason": f"deep_bear r={ratio:.2f}", "btc_ratio": ratio}
        elif ratio < 1.0:
            return {"vote": 1, "confidence": 0.7, "reason": f"bottom_recover r={ratio:.2f}", "btc_ratio": ratio}
        elif ratio <= 1.5:
            return {"vote": 1, "confidence": 0.6, "reason": f"healthy r={ratio:.2f}", "btc_ratio": ratio}
        else:
            return {"vote": 0, "confidence": 0.3, "reason": f"extended r={ratio:.2f}", "btc_ratio": ratio}


class TrendSignal:
    name = "trend"

    def __init__(self):
        self.spy_df = None
        self.vix_df = None
        self.xlk_df = None
        self.xlf_df = None

    def _load(self):
        if self.spy_df is None:
            self.spy_df = load_tv_data("SPY", "")
        if self.vix_df is None:
            self.vix_df = load_tv_data("VIX", "")
        if self.xlk_df is None:
            self.xlk_df = load_tv_data("XLK", "")
        if self.xlf_df is None:
            self.xlf_df = load_tv_data("XLF", "")

    def evaluate(self, date):
        self._load()
        conditions_met = 0
        reasons = []
        spy_sub = _safe_loc(self.spy_df, pd.Timestamp(date))
        if spy_sub is not None and len(spy_sub) >= 200:
            spy_close = spy_sub["close"]
            ema200 = float(_ema(spy_close, 200).iloc[-1])
            spy_price = float(spy_close.iloc[-1])
            if spy_price > ema200:
                conditions_met += 1
                reasons.append(f"spy>{ema200:.0f}")
        vix_sub = _safe_loc(self.vix_df, pd.Timestamp(date))
        if vix_sub is not None:
            vix = float(vix_sub["close"].iloc[-1])
            if vix < 25:
                conditions_met += 1
                reasons.append(f"vix={vix:.0f}")
        for sym, df in [("XLK", self.xlk_df), ("XLF", self.xlf_df)]:
            sub = _safe_loc(df, pd.Timestamp(date))
            if sub is not None and len(sub) >= 50:
                close = sub["close"]
                ema50 = float(_ema(close, 50).iloc[-1])
                price = float(close.iloc[-1])
                if price > ema50:
                    conditions_met += 1
                    reasons.append(f"{sym}>ema50")
                    break
        if conditions_met >= 2:
            vote = 1
        elif conditions_met == 0:
            vote = -1
        else:
            vote = 0
        return {"vote": vote, "confidence": conditions_met / 3, "reason": " ".join(reasons) + f" ({conditions_met}/3)", "conditions_met": conditions_met}


class SetupSignal:
    name = "setup"

    def find_setups(self, df, lookback=20):
        if len(df) < lookback + 5:
            return [], {}
        close = df["close"]
        high = df["high"]
        volume = df["volume"]
        last = float(close.iloc[-1])
        hi_20 = float(high.iloc[-lookback-1:-1].max())
        ma_20 = float(close.iloc[-lookback:].mean())
        ma_20_prev = float(close.iloc[-lookback-1:-1].mean())
        run_10d = float(close.iloc[-6] / close.iloc[-16] - 1) if len(close) > 16 else 0
        range_5d = (float(close.iloc[-5:].max()) - float(close.iloc[-5:].min())) / last if last > 0 else 0
        r = _rsi(close, 14)
        r_last = float(r.iloc[-1]) if len(r) > 0 else 50
        r_prev = float(r.iloc[-2]) if len(r) > 1 else 50
        vol_avg = float(volume.iloc[-lookback:].mean())
        vol_now = float(volume.iloc[-1])
        vol_ratio = vol_now / vol_avg if vol_avg > 0 else 1
        tags = []
        if last > hi_20 and vol_ratio > 1.5 and 40 <= r_last <= 70:
            tags.append("breakout_20d")
        if run_10d > 0.10 and range_5d < 0.05:
            tags.append("bull_flag")
        if float(close.iloc[-2]) < ma_20_prev and last > ma_20:
            tags.append("reclaim_ma20")
        if r_last < 30:
            tags.append("oversold")
        if r_prev < 30 and last > float(close.iloc[-2]):
            tags.append("oversold_bounce")
        metrics = {"rsi": r_last, "high_20": hi_20, "ma_20": ma_20, "vol_ratio": vol_ratio, "price": last}
        return tags, metrics

    def score(self, tags, metrics):
        s = 0
        if "breakout_20d" in tags: s += 3
        if "bull_flag" in tags: s += 2
        if "reclaim_ma20" in tags: s += 2
        if "oversold_bounce" in tags: s += 2
        if "oversold" in tags: s += 1
        if metrics.get("vol_ratio", 0) > 2.0: s += 1
        return s

    def evaluate(self, df_to_date, min_score=4):
        if df_to_date is None or len(df_to_date) < 25:
            return {"vote": 0, "confidence": 0, "reason": "no_history", "tags": [], "score": 0}
        tags, metrics = self.find_setups(df_to_date)
        if not tags:
            return {"vote": 0, "confidence": 0, "reason": "no_tags", "tags": [], "score": 0, "metrics": metrics}
        s = self.score(tags, metrics)
        vote = 1 if s >= min_score else 0
        return {"vote": vote, "confidence": min(1.0, s / 8), "reason": f"tags={tags} score={s}", "tags": tags, "score": s, "metrics": metrics}


class MultiSignalEngine:

    CRYPTO_SYMBOLS = {"BTCUSD", "BTCUSDT", "ETHUSD", "ETHUSDT", "SOLUSD",
                      "SOLUSDT", "BNBUSD", "BNBUSDT", "XRPUSD", "XRPUSDT",
                      "ADAUSD", "ADAUSDT", "DOGEUSDT", "DOTUSDT", "LINKUSDT",
                      "AVAXUSDT"}

    def __init__(self, min_votes_to_buy=3):
        self.regime = RegimeSignal()
        self.sentiment = SentimentSignal()
        self.onchain = OnchainSignal()
        self.trend = TrendSignal()
        self.setup = SetupSignal()
        self.min_votes = min_votes_to_buy

    def _is_crypto(self, symbol):
        return symbol in self.CRYPTO_SYMBOLS or "USD" in symbol or "USDT" in symbol

    def evaluate_macro(self, date):
        r = self.regime.evaluate(date)
        s = self.sentiment.evaluate(date)
        t = self.trend.evaluate(date)
        return {"regime": r, "sentiment": s, "trend": t}

    def evaluate_asset(self, date, asset_df, symbol, macro_votes):
        sub = _safe_loc(asset_df, pd.Timestamp(date))
        if sub is None or len(sub) < 25:
            return {"decision": "HOLD", "total": 0, "reason": "no_history", "votes": {}, "score": 0, "details": {}}
        setup_vote = self.setup.evaluate(sub)
        is_crypto = self._is_crypto(symbol)
        onchain_vote = self.onchain.evaluate(date, is_crypto=is_crypto)
        votes = {
            "regime": macro_votes["regime"]["vote"],
            "sentiment": macro_votes["sentiment"]["vote"],
            "trend": macro_votes["trend"]["vote"],
            "onchain": onchain_vote["vote"],
            "setup": setup_vote["vote"],
        }
        total = sum(votes.values())
        pos = sum(1 for v in votes.values() if v == 1)
        neg = sum(1 for v in votes.values() if v == -1)
        if total >= self.min_votes and pos >= 3:
            decision = "BUY"
        elif total <= -self.min_votes and neg >= 3:
            decision = "SELL"
        else:
            decision = "HOLD"
        return {"decision": decision, "total": total, "pos_votes": pos, "neg_votes": neg, "votes": votes, "details": {"regime": macro_votes["regime"], "sentiment": macro_votes["sentiment"], "trend": macro_votes["trend"], "onchain": onchain_vote, "setup": setup_vote}}
