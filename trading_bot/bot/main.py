import os
import sys
import time
import json
import yaml
import signal as sig_module
import pandas as pd
from datetime import datetime, timezone, timedelta

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from backtest.run_v8_hedge_fund import (
    UNIVERSE_V8, load_tv_data, load_macro_features,
    build_momentum_table, get_all_etf_scores, select_v8,
    compute_metrics, compute_spy_buyhold,
)
from features.options_hedge import (
    compute_market_score, score_to_regime, score_to_allocation,
    OptionsHedgeManager, black_scholes_put,
)
from models.ml_predictor import MLPredictor


def download_universe_data():
    try:
        from tvDatafeed import TvDatafeed, Interval
    except ImportError:
        return
    print("Descargando datos frescos...")
    try:
        tv = TvDatafeed()
    except Exception:
        return
    for sym, exch, desc in UNIVERSE_V8:
        cache_path = os.path.join(ROOT, "data", "tv_cache", f"{sym}_{exch}.csv")
        if True:
            try:
                df = tv.get_hist(symbol=sym, exchange=exch, interval=Interval.in_daily, n_bars=5000)
                if df is not None and len(df) > 0:
                    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
                    df.to_csv(cache_path)
                    print(f"  {sym}: {len(df)} barras")
            except Exception:
                pass
    vix_path = os.path.join(ROOT, "data", "tv_cache", "VIX_CBOE.csv")
    if True:
        try:
            df = tv.get_hist(symbol="VIX", exchange="CBOE", interval=Interval.in_daily, n_bars=5000)
            if df is not None and len(df) > 0:
                df.to_csv(vix_path)
        except Exception:
            pass
    print("Datos descargados.")


class TradingBotV8:

    def __init__(self, config_path=None):
        if config_path is None:
            config_path = os.path.join(ROOT, "config.yaml")
        if os.path.exists(config_path):
            with open(config_path) as f:
                self.config = yaml.safe_load(f)
        else:
            self.config = {}
        self.brokers = {}
        self.positions = {}
        self.running = False
        self.log_path = os.path.join(ROOT, "results", "live_log_v8.txt")
        self.state_path = os.path.join(ROOT, "results", "live_state_v8.json")
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
        os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
        self.last_rebalance = None
        self.state = self._load_state()
        self.hedge_manager = OptionsHedgeManager()
        self.ml_predictor = MLPredictor()
        self.stop_loss_pct = -0.05
        self.trailing_stop_trigger = 0.10
        self.trailing_stop_drop = -0.05

    def log(self, msg):
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        line = f"[{ts}] {msg}"
        print(line)
        try:
            with open(self.log_path, "a") as f:
                f.write(line + "\n")
        except Exception:
            pass

    def _load_state(self):
        if os.path.exists(self.state_path):
            try:
                with open(self.state_path) as f:
                    return json.load(f)
            except Exception:
                pass
        return {"positions": {}, "cash": 500.0, "history": [], "last_rebalance": None}

    def _save_state(self):
        try:
            with open(self.state_path, "w") as f:
                json.dump(self.state, f, indent=2, default=str)
        except Exception as e:
            self.log(f"ERROR guardando estado: {e}")

    def connect_brokers(self):
        self.log("Modo: PAPER TRADING SIMULADO")
        self.log(f"Capital inicial: ${self.state.get('cash', 500.0):.2f}")
        self.log(f"Posiciones actuales: {list(self.state.get('positions', {}).keys())}")

    def is_rebalance_day(self):
        today = datetime.now(timezone.utc).date()
        if today.day <= 3 and today.weekday() < 5:
            last = self.state.get("last_rebalance")
            if last:
                last_date = datetime.fromisoformat(last.replace('Z', '+00:00')).date()
                if last_date.month == today.month and last_date.year == today.year:
                    return False
            return True
        return False

    def has_positions(self):
        positions = self.state.get("positions", {})
        return len(positions) > 0

    def run_rebalance(self):
        download_universe_data()
        self.log("=" * 60)
        self.log("REBALANCEO - Bot v12")
        self.log("=" * 60)
        self.log("Cargando features macro...")
        try:
            features = load_macro_features()
            if not features.empty:
                last_feat = features.iloc[-1]
                self.log(f"  VIX={last_feat.get('vix', 'n/a')}")
        except Exception as e:
            self.log(f"ERROR cargando features macro: {e}")
            features = pd.DataFrame()
        self.log("Calculando momentum...")
        try:
            mom_df = build_momentum_table(UNIVERSE_V8)
            if mom_df.empty:
                self.log("ERROR: sin datos de momentum")
                return
        except Exception as e:
            self.log(f"ERROR calculando momentum: {e}")
            return
        self.log("Cargando scores fundamentales...")
        try:
            etf_scores = get_all_etf_scores()
        except Exception as e:
            self.log(f"ERROR scores fundamentales: {e}")
            etf_scores = {}
        opt_config_path = os.path.join(ROOT, "results", "optimized_config.json")
        opt_config = {}
        if os.path.exists(opt_config_path):
            try:
                with open(opt_config_path) as f:
                    opt_config = json.load(f)
                self.log(f"Usando config optimizada: {opt_config}")
            except Exception:
                pass
        
        # === v12: ML PREDICTION ===
        ml_pred = 0
        ml_acc = 0
        try:
            self.log("Prediccion ML (LightGBM)...")
            spy_df = load_tv_data("SPY", "AMEX")
            if not spy_df.empty:
                spy_features = self.ml_predictor.prepare_features(spy_df, features)
                ml_pred, ml_acc = self.ml_predictor.train_predict(spy_features)
                self.log(f"  ML prediccion: {ml_pred} (1=sube, 0=flat, -1=baja) - accuracy: {ml_acc:.1%}")
        except Exception as e:
            self.log(f"  ML error: {e}")

        self.log("Seleccionando top-3...")
        try:
            weights, selections, cash_months = select_v8(mom_df, features, etf_scores, **opt_config)
            if not selections:
                self.log("ERROR: sin selecciones")
                return
        except Exception as e:
            self.log(f"ERROR en seleccion: {e}")
            return
        
        # Si ML predice bajista (-1), reducir exposure
        last_selection = selections[-1]
        if ml_pred == -1:
            self.log("  ML predice BAJISTA - reduciendo exposure 50%")
            last_selection["cash_pct"] = max(0.5, last_selection.get("cash_pct", 0))
        elif ml_pred == 1 and last_selection.get("cash_pct", 0) < 0.3:
            self.log("  ML predice ALCISTA - manteniendo exposure")
        
        self.log(f"Regimen: {last_selection.get('regime', 'NORMAL')}")
        self.log(f"Cash %: {last_selection.get('cash_pct', 0)*100:.1f}%")
        self.log(f"Seleccionados: {last_selection.get('selected', [])}")
        cash = self.state.get("cash", 500.0)
        current_positions = self.state.get("positions", {})
        current_value = sum(p.get("value", 0) for p in current_positions.values())
        total_capital = cash + current_value
        cash_pct = last_selection.get("cash_pct", 0)
        investable = total_capital * (1 - cash_pct)
        new_positions = {}
        if "weights" in last_selection:
            for asset, w in last_selection["weights"].items():
                if w > 0:
                    target_value = investable * w
                    df = load_tv_data(asset, "")
                    if not df.empty:
                        price = float(df["close"].iloc[-1])
                        qty = target_value / price if price > 0 else 0
                        new_positions[asset] = {
                            "qty": qty,
                            "entry_price": price,
                            "value": target_value,
                            "weight": w,
                            "peak_price": price,
                        }
                        self.log(f"  >>> {asset}: qty={qty:.6f} @ ${price:.2f} = ${target_value:.2f}")
        cash_remaining = total_capital - sum(p["value"] for p in new_positions.values())
        self.state["positions"] = new_positions
        self.state["cash"] = cash_remaining
        self.state["last_rebalance"] = datetime.now(timezone.utc).isoformat()
        self.state["ml_prediction"] = ml_pred
        self.state["ml_accuracy"] = ml_acc
        self.state["history"].append({
            "date": datetime.now(timezone.utc).isoformat(),
            "regime": last_selection.get("regime"),
            "selected": last_selection.get("selected"),
            "cash_pct": cash_pct,
            "total_value": total_capital,
            "ml_pred": ml_pred,
        })
        self._save_state()
        self.log(f"Estado guardado. Total: ${total_capital:.2f}")

    def daily_check(self):
        download_universe_data()
        positions = self.state.get("positions", {})
        if not positions:
            self.log("Sin posiciones. Total: $500.00 (cash)")
            return
        cash = self.state.get("cash", 0)
        total_value = cash
        updated_positions = {}
        positions_to_sell = []
        for asset, pos in positions.items():
            df = load_tv_data(asset, "")
            if not df.empty:
                current_price = float(df["close"].iloc[-1])
                qty = pos.get("qty", 0)
                entry_price = pos.get("entry_price", current_price)
                peak_price = pos.get("peak_price", entry_price)
                value = qty * current_price
                pnl = (current_price - entry_price) * qty
                pnl_pct = ((current_price / entry_price) - 1) * 100 if entry_price > 0 else 0
                total_value += value
                if current_price > peak_price:
                    peak_price = current_price
                if pnl_pct / 100 <= self.stop_loss_pct:
                    self.log(f"  STOP-LOSS {asset}: {pnl_pct:+.2f}% <= {self.stop_loss_pct*100:.0f}% - VENDIENDO")
                    positions_to_sell.append((asset, qty, current
