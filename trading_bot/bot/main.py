"""
Orquestador live paper trading - Bot v8/v9.

- Si no hay posiciones: invierte automaticamente
- Si hay posiciones: actualiza precios y deja crecer
- Dia 1 del mes: rebalancea (rota si hace falta)
"""
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


def download_universe_data():
    try:
        from tvDatafeed import TvDatafeed, Interval
    except ImportError:
        return
    print("Descargando datos frescos de TradingView...")
    try:
        tv = TvDatafeed()
    except Exception:
        return
    for sym, exch, desc in UNIVERSE_V8:
        cache_path = os.path.join(ROOT, "data", "tv_cache", f"{sym}_{exch}.csv")
        if not os.path.exists(cache_path):
            try:
                df = tv.get_hist(symbol=sym, exchange=exch, interval=Interval.in_daily, n_bars=5000)
                if df is not None and len(df) > 0:
                    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
                    df.to_csv(cache_path)
                    print(f"  {sym}: {len(df)} barras")
            except Exception:
                pass
    vix_path = os.path.join(ROOT, "data", "tv_cache", "VIX_CBOE.csv")
    if not os.path.exists(vix_path):
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
        self.log("REBALANCEO - Bot v8")
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

        self.log("Seleccionando top-3...")
        try:
            weights, selections, cash_months = select_v8(mom_df, features, etf_scores)
            if not selections:
                self.log("ERROR: sin selecciones")
                return
        except Exception as e:
            self.log(f"ERROR en seleccion: {e}")
            return

        last_selection = selections[-1]
        self.log(f"Regimen: {last_selection.get('regime', 'NORMAL')}")
        self.log(f"Cash %: {last_selection.get('cash_pct', 0)*100:.1f}%")
        self.log(f"Seleccionados: {last_selection.get('selected', [])}")

        # Calcular capital total (cash + valor posiciones actuales)
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
                        }
                        self.log(f"  >>> {asset}: qty={qty:.6f} @ ${price:.2f} = ${target_value:.2f}")

        cash_remaining = total_capital - sum(p["value"] for p in new_positions.values())
        self.state["positions"] = new_positions
        self.state["cash"] = cash_remaining
        self.state["last_rebalance"] = datetime.now(timezone.utc).isoformat()
        self.state["history"].append({
            "date": datetime.now(timezone.utc).isoformat(),
            "regime": last_selection.get("regime"),
            "selected": last_selection.get("selected"),
            "cash_pct": cash_pct,
            "total_value": total_capital,
        })

        self._save_state()
        self.log(f"Estado guardado. Total: ${total_capital:.2f}")

    def daily_check(self):
        """Actualiza precios de las posiciones y recalcula P&L."""
        positions = self.state.get("positions", {})
        if not positions:
            self.log("Sin posiciones. Total: $500.00 (cash)")
            return

        cash = self.state.get("cash", 0)
        total_value = cash
        updated_positions = {}

        for asset, pos in positions.items():
            df = load_tv_data(asset, "")
            if not df.empty:
                current_price = float(df["close"].iloc[-1])
                qty = pos.get("qty", 0)
                entry_price = pos.get("entry_price", current_price)
                value = qty * current_price
                pnl = (current_price - entry_price) * qty
                pnl_pct = ((current_price / entry_price) - 1) * 100 if entry_price > 0 else 0
                total_value += value

                # ACTUALIZAR el valor de la posicion con el precio actual
                updated_positions[asset] = {
                    "qty": qty,
                    "entry_price": entry_price,
                    "value": value,
                    "current_price": current_price,
                    "weight": pos.get("weight", 0),
                }

                self.log(f"  {asset}: ${current_price:.2f} x {qty:.6f} = ${value:.2f} (P&L {pnl:+.2f}, {pnl_pct:+.2f}%)")
            else:
                updated_positions[asset] = pos
                total_value += pos.get("value", 0)

        # Guardar posiciones actualizadas
        self.state["positions"] = updated_positions

        total_pnl = total_value - 500
        total_pnl_pct = (total_pnl / 500) * 100
        self.log(f"Total: ${total_value:.2f} (P&L {total_pnl:+.2f}, {total_pnl_pct:+.2f}%)")

        # Guardar snapshot diario
        snapshot = {
            "date": datetime.now(timezone.utc).isoformat(),
            "total_value": total_value,
            "cash": cash,
            "positions_count": len(updated_positions),
            "pnl": total_pnl,
            "pnl_pct": total_pnl_pct,
        }
        if "daily_snapshots" not in self.state:
            self.state["daily_snapshots"] = []
        self.state["daily_snapshots"].append(snapshot)
        if len(self.state["daily_snapshots"]) > 90:
            self.state["daily_snapshots"] = self.state["daily_snapshots"][-90:]

        self._save_state()

    def run_cycle(self):
        self.log("=" * 60)
        self.log("Ciclo bot v8")
        self.log("=" * 60)

        # LOGICA CLAVE:
        # 1. Si no hay posiciones -> INVERTIR (rebalancear para tomar posiciones)
        # 2. Si hay posiciones y es dia 1 del mes -> REBALANCEAR (rotar)
        # 3. Si hay posiciones y no es dia 1 -> SOLO ACTUALIZAR PRECIOS (dejar crecer)

        if not self.has_positions():
            self.log("Sin posiciones - invirtiendo automaticamente...")
            self.run_rebalance()
        elif self.is_rebalance_day():
            self.log("Es dia de rebalanceo mensual - rotando posiciones...")
            self.run_rebalance()
        else:
            self.log("Actualizando precios (dejando crecer)...")
            self.daily_check()

        self.log("Ciclo completado")

    def run(self, interval_seconds=3600):
        self.running = True
        self.log(f"Bot v8 iniciado. Ctrl-C para parar.")

        def handle_sig(signum, frame):
            self.running = False
        sig_module.signal(sig_module.SIGINT, handle_sig)
        sig_module.signal(sig_module.SIGTERM, handle_sig)

        while self.running:
            try:
                self.run_cycle()
            except Exception as e:
                self.log(f"ERROR en ciclo: {e}")
                import traceback
                traceback.print_exc()
            slept = 0
            while slept < interval_seconds and self.running:
                time.sleep(min(5, interval_seconds - slept))
                slept += 5

        self.log("Bot detenido")


if __name__ == "__main__":
    bot = TradingBotV8()
    bot.connect_brokers()
    if "--once" in sys.argv:
        bot.run_cycle()
    elif "--rebalance" in sys.argv:
        bot.run_rebalance()
    else:
        bot.run(interval_seconds=6 * 3600)
