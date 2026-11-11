import os
import sys
import time
import json
import yaml
import signal as sig_module
import pandas as pd
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from backtest.run_v8_hedge_fund import (
    UNIVERSE_V8, load_tv_data, load_macro_features,
    build_momentum_table, get_all_etf_scores, select_v8,
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
        try:
            df = tv.get_hist(symbol=sym, exchange=exch, interval=Interval.in_daily, n_bars=5000)
            if df is not None and len(df) > 0:
                os.makedirs(os.path.dirname(cache_path), exist_ok=True)
                df.to_csv(cache_path)
                print(f"  {sym}: {len(df)} barras")
        except Exception:
            pass
    try:
        df = tv.get_hist(symbol="VIX", exchange="CBOE", interval=Interval.in_daily, n_bars=5000)
        if df is not None and len(df) > 0:
            df.to_csv(os.path.join(ROOT, "data", "tv_cache", "VIX_CBOE.csv"))
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
        self.log_path = os.path.join(ROOT, "results", "live_log_v8.txt")
        self.state_path = os.path.join(ROOT, "results", "live_state_v8.json")
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
        os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
        self.state = self._load_state()
        self.hedge_manager = OptionsHedgeManager()
        self.ml_predictor = MLPredictor()
        self.stop_loss_pct = -0.05
        self.trailing_trigger = 0.10
        self.trailing_drop = -0.05

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
        return len(self.state.get("positions", {})) > 0

    def run_rebalance(self):
        download_universe_data()
        self.log("=" * 60)
        self.log("REBALANCEO - Bot v12")
        self.log("=" * 60)

        self.log("Cargando features macro...")
        try:
            features = load_macro_features()
        except Exception as e:
            self.log(f"ERROR features: {e}")
            features = pd.DataFrame()

        self.log("Calculando momentum...")
        try:
            mom_df = build_momentum_table(UNIVERSE_V8)
            if mom_df.empty:
                self.log("ERROR: sin momentum")
                return
        except Exception as e:
            self.log(f"ERROR momentum: {e}")
            return

        self.log("Cargando scores fundamentales...")
        try:
            etf_scores = get_all_etf_scores()
        except Exception as e:
            self.log(f"ERROR scores: {e}")
            etf_scores = {}

        opt_config = {}
        opt_path = os.path.join(ROOT, "results", "optimized_config.json")
        if os.path.exists(opt_path):
            try:
                with open(opt_path) as f:
                    opt_config = json.load(f)
                self.log(f"Config optimizada: {opt_config}")
            except Exception:
                pass

        ml_pred = 0
        ml_acc = 0
        try:
            self.log("Prediccion ML (LightGBM)...")
            spy_df = load_tv_data("SPY", "AMEX")
            if not spy_df.empty:
                spy_feat = self.ml_predictor.prepare_features(spy_df, features)
                ml_pred, ml_acc = self.ml_predictor.train_predict(spy_feat)
                self.log(f"  ML: pred={ml_pred} acc={ml_acc:.1%}")
        except Exception as e:
            self.log(f"  ML error: {e}")

        self.log("Seleccionando top-3...")
        try:
            weights, selections, cash_m = select_v8(mom_df, features, etf_scores, **opt_config)
            if not selections:
                self.log("ERROR: sin selecciones")
                return
        except Exception as e:
            self.log(f"ERROR seleccion: {e}")
            return

        last_sel = selections[-1]
        if ml_pred == -1:
            self.log("  ML predice BAJISTA - reduciendo exposure 50%")
            last_sel["cash_pct"] = max(0.5, last_sel.get("cash_pct", 0))

        self.log(f"Regimen: {last_sel.get('regime', 'NORMAL')}")
        self.log(f"Cash %: {last_sel.get('cash_pct', 0)*100:.1f}%")
        self.log(f"Seleccionados: {last_sel.get('selected', [])}")

        cash = self.state.get("cash", 500.0)
        cur_val = sum(p.get("value", 0) for p in self.state.get("positions", {}).values())
        total_cap = cash + cur_val
        cash_pct = last_sel.get("cash_pct", 0)
        investable = total_cap * (1 - cash_pct)

        new_pos = {}
        if "weights" in last_sel:
            for asset, w in last_sel["weights"].items():
                if w > 0:
                    target_val = investable * w
                    df = load_tv_data(asset, "")
                    if not df.empty:
                        price = float(df["close"].iloc[-1])
                        qty = target_val / price if price > 0 else 0
                        new_pos[asset] = {"qty": qty, "entry_price": price, "value": target_val, "weight": w, "peak_price": price}
                        self.log(f"  >>> {asset}: qty={qty:.6f} @ ${price:.2f} = ${target_val:.2f}")

        cash_rem = total_cap - sum(p["value"] for p in new_pos.values())
        self.state["positions"] = new_pos
        self.state["cash"] = cash_rem
        self.state["last_rebalance"] = datetime.now(timezone.utc).isoformat()
        self.state["ml_prediction"] = ml_pred
        self.state["ml_accuracy"] = ml_acc
        self.state["history"].append({
            "date": datetime.now(timezone.utc).isoformat(),
            "regime": last_sel.get("regime"),
            "selected": last_sel.get("selected"),
            "cash_pct": cash_pct,
            "total_value": total_cap,
            "ml_pred": ml_pred,
        })
        self._save_state()
        self.log(f"Estado guardado. Total: ${total_cap:.2f}")

    def daily_check(self):
        download_universe_data()
        positions = self.state.get("positions", {})
        if not positions:
            self.log("Sin posiciones. Total: $500.00 (cash)")
            return

        cash = self.state.get("cash", 0)
        total_value = cash
        updated = {}
        to_sell = []

        for asset, pos in positions.items():
            df = load_tv_data(asset, "")
            if not df.empty:
                cur_price = float(df["close"].iloc[-1])
                qty = pos.get("qty", 0)
                entry = pos.get("entry_price", cur_price)
                peak = pos.get("peak_price", entry)
                value = qty * cur_price
                pnl = (cur_price - entry) * qty
                pnl_pct = ((cur_price / entry) - 1) * 100 if entry > 0 else 0
                total_value += value

                if cur_price > peak:
                    peak = cur_price

                if pnl_pct / 100 <= self.stop_loss_pct:
                    self.log(f"  STOP-LOSS {asset}: {pnl_pct:+.2f}% - VENDIENDO")
                    to_sell.append((asset, qty, cur_price, pnl, "STOP_LOSS"))
                    continue

                gain = (peak / entry - 1) if entry > 0 else 0
                drop = (cur_price / peak - 1) if peak > 0 else 0
                if gain >= self.trailing_trigger and drop <= self.trailing_drop:
                    self.log(f"  TRAILING STOP {asset} - VENDIENDO")
                    to_sell.append((asset, qty, cur_price, pnl, "TRAILING_STOP"))
                    continue

                updated[asset] = {"qty": qty, "entry_price": entry, "value": value, "current_price": cur_price, "weight": pos.get("weight", 0), "peak_price": peak}
                self.log(f"  {asset}: ${cur_price:.2f} x {qty:.6f} = ${value:.2f} (P&L {pnl:+.2f}, {pnl_pct:+.2f}%)")
            else:
                updated[asset] = pos
                total_value += pos.get("value", 0)

        if to_sell:
            for asset, qty, sell_price, pnl, reason in to_sell:
                cash += qty * sell_price
                self.log(f"  >>> VENDIDO {asset} @ ${sell_price:.2f} (P&L {pnl:+.2f}) - {reason}")
                if "trade_history" not in self.state:
                    self.state["trade_history"] = []
                self.state["trade_history"].append({"symbol": asset, "exit_price": sell_price, "qty": qty, "pnl": pnl, "exit_reason": reason, "date": datetime.now(timezone.utc).isoformat()})

        self.state["positions"] = updated
        self.state["cash"] = cash
        total_pnl = total_value - 500
        total_pnl_pct = (total_pnl / 500) * 100
        self.log(f"Total: ${total_value:.2f} (P&L {total_pnl:+.2f}, {total_pnl_pct:+.2f}%)")

        if to_sell and cash > 50:
            self.log("  REINVERTIR: rebalanceando...")
            self.run_rebalance()
            return

        try:
            features = load_macro_features()
            if not features.empty:
                lf = features.iloc[-1]
                vix = lf.get("vix", 25)
                fg = lf.get("fear_greed", 50)
                ff = lf.get("fed_funds", 4.5)
                yinv = bool(lf.get("yield_inverted", 0))
            else:
                vix, fg, ff, yinv = 25, 50, 4.5, False

            spy_df = load_tv_data("SPY", "AMEX")
            spy_above = True
            spy_price = 500.0
            spy_mom = 0
            if not spy_df.empty:
                spy_price = float(spy_df["close"].iloc[-1])
                ema200 = spy_df["close"].rolling(200).mean().iloc[-1]
                spy_above = spy_price > float(ema200) if not pd.isna(ema200) else True
                spy_mom = float(spy_df["close"].pct_change(20).iloc[-1] * 100) if len(spy_df) > 20 else 0

            score = compute_market_score(vix, fg, spy_above, yinv, ff, spy_mom, spy_mom * 2)
            regime = score_to_regime(score)
            eq_pct, cash_pct, hedge_pct = score_to_allocation(score)
            self.log(f"Market Score: {score:+.0f} ({regime}) -> {eq_pct*100:.0f}% eq / {cash_pct*100:.0f}% cash / {hedge_pct*100:.0f}% hedge")

            if score < -20 and len(self.hedge_manager.positions) == 0:
                hedge_budget = total_value * hedge_pct
                put_price = black_scholes_put(spy_price, spy_price * 0.95, 30/365, 0.045, vix / 100)
                if put_price > 0:
                    n_contracts = max(1, int(hedge_budget / (put_price * 100)))
                    pos = self.hedge_manager.buy_put(spy_price, vix, quantity=n_contracts)
                    self.log(f"  HEDGE: Comprados {n_contracts} puts SPY @ ${put_price:.2f}")

            if self.hedge_manager.positions:
                self.hedge_manager.update_positions(spy_price, vix)
                hs = self.hedge_manager.get_summary()
                self.log(f"  HEDGE: {hs['open_positions']} puts, P&L ${hs['pnl']:.2f}")

            if score > 0 and self.hedge_manager.positions:
                self.hedge_manager.close_all(spy_price, vix)
                self.log("  HEDGE: Cerrados todos los puts")

            self.state["market_score"] = score
            self.state["market_regime"] = regime
            self.state["hedge_summary"] = self.hedge_manager.get_summary()
        except Exception as e:
            self.log(f"  HEDGE error: {e}")

        snapshot = {"date": datetime.now(timezone.utc).isoformat(), "total_value": total_value, "cash": cash, "pnl": total_pnl, "pnl_pct": total_pnl_pct}
        if "daily_snapshots" not in self.state:
            self.state["daily_snapshots"] = []
        self.state["daily_snapshots"].append(snapshot)
        if len(self.state["daily_snapshots"]) > 90:
            self.state["daily_snapshots"] = self.state["daily_snapshots"][-90:]
        self._save_state()

    def run_cycle(self):
        self.log("=" * 60)
        self.log("Ciclo bot v12")
        self.log("=" * 60)
        if not self.has_positions():
            self.log("Sin posiciones - invirtiendo...")
            self.run_rebalance()
        elif self.is_rebalance_day():
            self.log("Rebalanceo mensual...")
            self.run_rebalance()
        else:
            self.log("Actualizando precios...")
            self.daily_check()
        self.log("Ciclo completado")

    def run(self, interval_seconds=3600):
        self.running = True
        self.log("Bot v12 iniciado.")
        def handle_sig(signum, frame):
            self.running = False
        sig_module.signal(sig_module.SIGINT, handle_sig)
        sig_module.signal(sig_module.SIGTERM, handle_sig)
        while self.running:
            try:
                self.run_cycle()
            except Exception as e:
                self.log(f"ERROR: {e}")
                import traceback
                traceback.print_exc()
            slept = 0
            while slept < interval_seconds and self.running:
                time.sleep(min(5, interval_seconds - slept))
                slept += 5
        self.log("Bot detenido")


if __name__ == "__main__":
    bot = TradingBotV8()
    if "--once" in sys.argv:
        bot.run_cycle()
    elif "--rebalance" in sys.argv:
        bot.run_rebalance()
    else:
        bot.run(interval_seconds=6 * 3600)
