"""
Orquestador live paper trading - Bot v8/v9.

Estrategia: Sector Rotation Top-3 mensual con:
  - Momentum multi-lookback (3m/6m/12m)
  - VIX rules + FedFunds + yield curve (regime-aware)
  - Agentes ai-hedge-fund (Graham/Buffett/Munger/Lynch/Druckenmiller)
  - SEC EDGAR fundamentales (sin API key)
  - Anomaly detection (Metabase modified z-score)
  - Anti-bubble + caps
  - v9: stop-loss + crypto winter + smart money exit

Ciclo: cada barra (diaria por defecto), revisa si toca rebalancear.
Rebalanceo mensual: primer dia habil del mes.
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
def download_universe_data():
    """Descarga datos frescos de TradingView si no existen en cache."""
    from tvDatafeed import TvDatafeed, Interval
    print("Descargando datos frescos de TradingView...")
    tv = TvDatafeed()
    for sym, exch, desc in UNIVERSE_V8:
        cache_path = os.path.join(ROOT, "data", "tv_cache", f"{sym}_{exch}.csv")
        if not os.path.exists(cache_path):
            try:
                df = tv.get_hist(symbol=sym, exchange=exch, interval=Interval.in_daily, n_bars=5000)
                if df is not None and len(df) > 0:
                    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
                    df.to_csv(cache_path)
                    print(f"  {sym}: {len(df)} barras descargadas")
            except Exception as e:
                print(f"  {sym}: ERROR {e}")
    # Tambien descargar VIX
    vix_path = os.path.join(ROOT, "data", "tv_cache", "VIX_CBOE.csv")
    if not os.path.exists(vix_path):
        try:
            df = tv.get_hist(symbol="VIX", exchange="CBOE", interval=Interval.in_daily, n_bars=5000)
            if df is not None and len(df) > 0:
                df.to_csv(vix_path)
                print(f"  VIX: {len(df)} barras descargadas")
        except Exception as e:
            print(f"  VIX: ERROR {e}")
    print("Datos descargados.")    
UNIVERSE_V8, load_tv_data, load_macro_features,
    build_momentum_table, get_all_etf_scores, select_v8,
    compute_metrics, compute_spy_buyhold,
)


class TradingBotV8:
    """Bot v8 para paper trading live."""

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
        self.last_rebalance = None
        self.state = self._load_state()

    def log(self, msg):
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        line = f"[{ts}] {msg}"
        print(line)
        with open(self.log_path, "a") as f:
            f.write(line + "\n")

    def _load_state(self):
        if os.path.exists(self.state_path):
            with open(self.state_path) as f:
                return json.load(f)
        return {"positions": {}, "cash": 500.0, "history": [], "last_rebalance": None}

    def _save_state(self):
        with open(self.state_path, "w") as f:
            json.dump(self.state, f, indent=2, default=str)

    def connect_brokers(self):
        """Conecta brokers paper. Por ahora: paper trading simulado."""
        self.log("Modo: PAPER TRADING SIMULADO (sin broker real)")
        self.log(f"Capital inicial: ${self.state.get('cash', 500.0):.2f}")
        self.log(f"Posiciones actuales: {self.state.get('positions', {})}")

    def is_rebalance_day(self):
        """Rebalancear el primer dia habil del mes."""
        today = datetime.now(timezone.utc).date()
        if today.day <= 3 and today.weekday() < 5:
            last = self.state.get("last_rebalance")
            if last:
                last_date = datetime.fromisoformat(last.replace('Z', '+00:00')).date()
                if last_date.month == today.month and last_date.year == today.year:
                    return False
            return True
        return False

    def run_rebalance(self):
        """Ejecuta rebalanceo mensual completo."""
        self.log("=" * 60)
        self.log("REBALANCEO MENSUAL - Bot v8")
        self.log("=" * 60)

        self.log("Cargando features macro (VIX, FedFunds, yield curve)...")
        features = load_macro_features()
        if not features.empty:
            last_feat = features.iloc[-1]
            self.log(f"  VIX={last_feat.get('vix', 'n/a')}, "
                     f"FedFunds={last_feat.get('fed_funds', 'n/a')}, "
                     f"yield_spread={last_feat.get('yield_spread', 'n/a')}")

        self.log("Calculando momentum multi-lookback...")
        mom_df = build_momentum_table(UNIVERSE_V8)
        if mom_df.empty:
            self.log("ERROR: sin datos de momentum")
            return

        self.log("Cargando scores fundamentales (agentes hedge fund)...")
        etf_scores = get_all_etf_scores()

        self.log("Seleccionando top-3 con regime-aware...")
        weights, selections, cash_months = select_v8(mom_df, features, etf_scores)
        if not selections:
            self.log("ERROR: sin selecciones")
            return

        last_selection = selections[-1]
        self.log(f"Mes actual: {last_selection['month'].strftime('%Y-%m')}")
        self.log(f"Regimen: {last_selection.get('regime', 'NORMAL')}")
        self.log(f"Razones: {last_selection.get('reasons', [])}")
        self.log(f"Cash %: {last_selection.get('cash_pct', 0)*100:.1f}%")
        self.log(f"Seleccionados: {last_selection.get('selected', [])}")
        if "weights" in last_selection:
            for asset, w in last_selection["weights"].items():
                self.log(f"  {asset}: {w*100:.1f}%")

        cash = self.state.get("cash", 500.0) + sum(p.get("value", 0) for p in self.state.get("positions", {}).values())
        cash_pct = last_selection.get("cash_pct", 0)
        investable = cash * (1 - cash_pct)

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

        cash_remaining = cash - sum(p["value"] for p in new_positions.values())
        self.state["positions"] = new_positions
        self.state["cash"] = cash_remaining
        self.state["last_rebalance"] = datetime.now(timezone.utc).isoformat()
        self.state["history"].append({
            "date": datetime.now(timezone.utc).isoformat(),
            "regime": last_selection.get("regime"),
            "selected": last_selection.get("selected"),
            "cash_pct": cash_pct,
            "total_value": cash,
        })
        self._save_state()
        self.log(f"Total portfolio: ${cash:.2f} (cash: ${cash_remaining:.2f}, invertido: ${cash - cash_remaining:.2f})")

    def daily_check(self):
        """Chequeo diario: actualizar precios, P&L, detectar anomalies."""
        positions = self.state.get("positions", {})
        if not positions:
            return

        total_value = self.state.get("cash", 0)
        for asset, pos in positions.items():
            df = load_tv_data(asset, "")
            if not df.empty:
                current_price = float(df["close"].iloc[-1])
                qty = pos.get("qty", 0)
                value = qty * current_price
                pnl = (current_price - pos.get("entry_price", 0)) * qty
                pnl_pct = (current_price / pos.get("entry_price", 1) - 1) * 100 if pos.get("entry_price") else 0
                total_value += value
                self.log(f"  {asset}: ${current_price:.2f} x {qty:.6f} = ${value:.2f} (P&L {pnl:+.2f}, {pnl_pct:+.2f}%)")

        self.log(f"Total portfolio value: ${total_value:.2f}")

        snapshot = {
            "date": datetime.now(timezone.utc).isoformat(),
            "total_value": total_value,
            "cash": self.state.get("cash", 0),
            "positions_count": len(positions),
        }
        if "daily_snapshots" not in self.state:
            self.state["daily_snapshots"] = []
        self.state["daily_snapshots"].append(snapshot)
        if len(self.state["daily_snapshots"]) > 90:
            self.state["daily_snapshots"] = self.state["daily_snapshots"][-90:]
        self._save_state()

    def run_cycle(self):
        """Un ciclo: chequeo diario + rebalanceo si toca."""
        self.log("=" * 60)
        self.log("Ciclo bot v8")
        self.log("=" * 60)

        if self.is_rebalance_day():
            self.log("Es dia de rebalanceo mensual.")
            self.run_rebalance()
        else:
            self.log("No es dia de rebalanceo. Chequeo diario.")
        self.daily_check()
        self.log("Ciclo completado")

    def run(self, interval_seconds=3600):
        """Loop infinito hasta Ctrl-C."""
        self.running = True
        self.log(f"Bot v8 iniciado. Intervalo: {interval_seconds}s. Ctrl-C para parar.")
        self.log(f"Log: {self.log_path}")
        self.log(f"State: {self.state_path}")

        def handle_sig(signum, frame):
            self.log("Senal de parada recibida. Cerrando...")
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
