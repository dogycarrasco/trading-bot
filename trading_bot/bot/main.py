"""
Orquestador live paper trading - Bot v13 EVENT-DRIVEN.

Diferencia vs v8 (mensual):
  - NO hay rebalanceo programado.
  - Cada ciclo: actualiza precios -> verifica stops -> busca setups -> compra solo si hay senal.
  - Vende SOLO por stop-loss (-5%), trailing stop (+10% peak, -5% drop) o fin de backtest.

Setups detectados (trader-scanner find_setups):
  - breakout_20d: rompe maximo 20 barras con volumen >1.5x
  - bull_flag: momentum 10d >10% + rango 5d <5%
  - reclaim_ma20: vuelta arriba de MA20
  - oversold: RSI <30
  - oversold_bounce: RSI <30 ayer + cierre arriba hoy

Filtros anti-overtrading (mejoras vs backtest v13 original):
  - Volumen >1.5x (no 1.0x) para validar breakouts
  - RSI 40-70 para breakouts (excluye sobrecomprados)
  - Cooldown 5 dias tras stop-loss en el mismo activo
  - Score minimo 4 para comprar
  - Max 5 posiciones simultaneas
  - Position size 33% del cash disponible

Estado persistente: results/live_state_v8.json (compatible con dashboard).
Log: results/live_log_v8.txt.
"""
import os
import sys
import time
import json
import yaml
import signal as sig_module
import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from backtest.run_v8_hedge_fund import (
    UNIVERSE_V8, load_tv_data, load_macro_features,
    compute_metrics, compute_spy_buyhold,
)


def rsi(close, period=14):
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1/period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1/period, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(50)


def find_setups(df, lookback=20):
    if len(df) < lookback + 5:
        return [], {}

    close = df["close"]
    high = df["high"]
    low = df["low"]
    volume = df["volume"]
    last = float(close.iloc[-1])

    hi_20 = float(high.iloc[-lookback-1:-1].max())
    ma_20 = float(close.iloc[-lookback:].mean())
    ma_20_prev = float(close.iloc[-lookback-1:-1].mean())

    run_10d = float(close.iloc[-6] / close.iloc[-16] - 1) if len(close) > 16 else 0
    range_5d = (float(close.iloc[-5:].max()) - float(close.iloc[-5:].min())) / last if last > 0 else 0

    r = rsi(close, 14)
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

    metrics = {
        "rsi": r_last,
        "high_20": hi_20,
        "ma_20": ma_20,
        "run_10d_pct": run_10d * 100,
        "range_5d_pct": range_5d * 100,
        "vol_ratio": vol_ratio,
        "price": last,
    }
    return tags, metrics


def score_setups(tags, metrics):
    score = 0
    if "breakout_20d" in tags: score += 3
    if "bull_flag" in tags: score += 2
    if "reclaim_ma20" in tags: score += 2
    if "oversold_bounce" in tags: score += 2
    if "oversold" in tags: score += 1
    if metrics.get("vol_ratio", 0) > 2.0: score += 1
    return score


class TradingBotV13:
    STOP_LOSS_PCT = -0.05
    TRAILING_TRIGGER = 0.10
    TRAILING_DROP = -0.05
    MAX_POSITIONS = 5
    POSITION_SIZE_PCT = 0.33
    COOLDOWN_DAYS = 5
    MIN_SCORE = 4
    MIN_TRADE_USD = 10

    def __init__(self, config_path=None):
        if config_path is None:
            config_path = os.path.join(ROOT, "config.yaml")
        if os.path.exists(config_path):
            with open(config_path) as f:
                self.config = yaml.safe_load(f)
        else:
            self.config = {}
        self.brokers = {}
        self.running = False
        self.log_path = os.path.join(ROOT, "results", "live_log_v8.txt")
        self.state_path = os.path.join(ROOT, "results", "live_state_v8.json")
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
        self.state = self._load_state()
        self.state.setdefault("version", "v13_event_driven")
        self.state.setdefault("positions", {})
        self.state.setdefault("cash", 500.0)
        self.state.setdefault("history", [])
        self.state.setdefault("daily_snapshots", [])
        self.state.setdefault("cooldowns", {})

    def log(self, msg):
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        line = f"[{ts}] {msg}"
        print(line)
        with open(self.log_path, "a") as f:
            f.write(line + "\n")

    def _load_state(self):
        if os.path.exists(self.state_path):
            try:
                with open(self.state_path) as f:
                    return json.load(f)
            except Exception:
                pass
        return {"version": "v13_event_driven",
                "positions": {}, "cash": 500.0,
                "history": [], "daily_snapshots": [],
                "cooldowns": {}}

    def _save_state(self):
        with open(self.state_path, "w") as f:
            json.dump(self.state, f, indent=2, default=str)

    def connect_brokers(self):
        self.log("Modo: PAPER TRADING SIMULADO (sin broker real)")
        self.log(f"Capital inicial: ${self.state.get('cash', 500.0):.2f}")
        self.log(f"Posiciones actuales: {len(self.state.get('positions', {}))}")

    def _today(self):
        return datetime.now(timezone.utc)

    def _cooldown_active(self, symbol):
        cd = self.state.get("cooldowns", {})
        until_iso = cd.get(symbol)
        if not until_iso:
            return False
        try:
            until = datetime.fromisoformat(until_iso.replace("Z", "+00:00"))
            return self._today() < until
        except Exception:
            return False

    def _set_cooldown(self, symbol):
        if "cooldowns" not in self.state:
            self.state["cooldowns"] = {}
        self.state["cooldowns"][symbol] = (self._today() + timedelta(days=self.COOLDOWN_DAYS)).isoformat()

    def update_positions_prices(self):
        positions = self.state.get("positions", {})
        if not positions:
            return
        for sym, pos in positions.items():
            df = load_tv_data(sym, "")
            if df.empty:
                continue
            price = float(df["close"].iloc[-1])
            pos["current_price"] = price
            peak = pos.get("peak_price", pos.get("entry_price", price))
            if price > peak:
                pos["peak_price"] = price
            pos["value"] = pos.get("qty", 0) * price
            entry = pos.get("entry_price", price)
            pnl = (price - entry) * pos.get("qty", 0)
            pnl_pct = (price / entry - 1) * 100 if entry > 0 else 0
            self.log(f"  {sym}: ${price:.2f} x {pos.get('qty', 0):.6f} = ${pos['value']:.2f} "
                     f"(P&L ${pnl:+.2f}, {pnl_pct:+.2f}%)")

    def check_stops(self):
        positions = self.state.get("positions", {})
        if not positions:
            return
        to_sell = []
        for sym, pos in positions.items():
            entry = pos.get("entry_price", 0)
            current = pos.get("current_price", entry)
            peak = pos.get("peak_price", entry)
            pnl_pct = (current / entry - 1) if entry > 0 else 0
            gain = (peak / entry - 1) if entry > 0 else 0
            drop = (current / peak - 1) if peak > 0 else 0

            reason = None
            if pnl_pct <= self.STOP_LOSS_PCT:
                reason = "STOP_LOSS"
            elif gain >= self.TRAILING_TRIGGER and drop <= self.TRAILING_DROP:
                reason = "TRAILING_STOP"

            if reason:
                to_sell.append((sym, reason))

        for sym, reason in to_sell:
            pos = positions[sym]
            price = pos.get("current_price", 0)
            entry = pos.get("entry_price", 0)
            qty = pos.get("qty", 0)
            pnl = (price - entry) * qty
            proceeds = qty * price
            self.state["cash"] = self.state.get("cash", 0) + proceeds
            self.log(f"  VENTA {sym}: {reason} @ ${price:.2f} (entry ${entry:.2f}) "
                     f"P&L ${pnl:+.2f} -> cash ${self.state['cash']:.2f}")
            if reason == "STOP_LOSS":
                self._set_cooldown(sym)
                self.log(f"  Cooldown {sym} activado {self.COOLDOWN_DAYS} dias")
            self.state["history"].append({
                "date": self._today().isoformat(),
                "action": "SELL",
                "symbol": sym,
                "price": price,
                "qty": qty,
                "pnl": pnl,
                "reason": reason,
            })
            del positions[sym]
        if to_sell:
            self._save_state()

    def scan_opportunities(self):
        positions = self.state.get("positions", {})
        cash = self.state.get("cash", 0)
        if cash < self.MIN_TRADE_USD:
            self.log(f"  Cash ${cash:.2f} < min ${self.MIN_TRADE_USD}. No se compra.")
            return
        if len(positions) >= self.MAX_POSITIONS:
            self.log(f"  {len(positions)} posiciones = max {self.MAX_POSITIONS}. No se compra.")
            return

        opportunities = []
        for sym, _, _ in UNIVERSE_V8:
            if sym in positions:
                continue
            if self._cooldown_active(sym):
                self.log(f"  {sym} en cooldown, skip")
                continue
            df = load_tv_data(sym, "")
            if df.empty or len(df) < 25:
                continue
            tags, metrics = find_setups(df)
            if not tags:
                continue
            score = score_setups(tags, metrics)
            if score < self.MIN_SCORE:
                continue
            opportunities.append({
                "symbol": sym,
                "tags": tags,
                "score": score,
                "price": metrics["price"],
                "rsi": metrics["rsi"],
                "vol_ratio": metrics["vol_ratio"],
            })

        if not opportunities:
            self.log("  Sin oportunidades con score >= %d" % self.MIN_SCORE)
            return

        opportunities.sort(key=lambda x: -x["score"])
        slots = self.MAX_POSITIONS - len(positions)
        bought = []
        for opp in opportunities[:slots]:
            amount = min(self.state["cash"] * self.POSITION_SIZE_PCT, self.state["cash"])
            if amount < self.MIN_TRADE_USD:
                break
            qty = amount / opp["price"]
            self.state["positions"][opp["symbol"]] = {
                "qty": qty,
                "entry_price": opp["price"],
                "current_price": opp["price"],
                "peak_price": opp["price"],
                "value": amount,
                "tags": opp["tags"],
                "entry_date": self._today().isoformat(),
            }
            self.state["cash"] -= amount
            bought.append(opp)
            self.log(f"  COMPRA {opp['symbol']}: score {opp['score']} tags {opp['tags']} "
                     f"@ ${opp['price']:.2f} qty {qty:.6f} = ${amount:.2f}")
            self.state["history"].append({
                "date": self._today().isoformat(),
                "action": "BUY",
                "symbol": opp["symbol"],
                "price": opp["price"],
                "qty": qty,
                "tags": opp["tags"],
                "score": opp["score"],
            })
        if bought:
            self._save_state()

    def snapshot(self):
        cash = self.state.get("cash", 0)
        total_value = cash + sum(p.get("value", 0) for p in self.state.get("positions", {}).values())
        self.log(f"Total portfolio value: ${total_value:.2f} "
                 f"(cash ${cash:.2f}, invertido ${total_value - cash:.2f})")
        snap = {
            "date": self._today().isoformat(),
            "total_value": total_value,
            "cash": cash,
            "positions_count": len(self.state.get("positions", {})),
            "positions": {k: {"value": v.get("value", 0),
                              "pnl_pct": (v.get("current_price", 0) / v.get("entry_price", 1) - 1) * 100
                              if v.get("entry_price", 0) > 0 else 0}
                          for k, v in self.state.get("positions", {}).items()},
        }
        snaps = self.state.setdefault("daily_snapshots", [])
        snaps.append(snap)
        if len(snaps) > 90:
            self.state["daily_snapshots"] = snaps[-90:]
        self.state["total_value"] = total_value
        self.state["last_update"] = self._today().isoformat()
        self._save_state()

    def run_cycle(self):
        self.log("=" * 60)
        self.log("Ciclo bot v13 EVENT-DRIVEN")
        self.log("=" * 60)
        self.log("Paso 1/3: Actualizar precios de posiciones")
        self.update_positions_prices()
        self.log("Paso 2/3: Verificar stops (stop-loss + trailing)")
        self.check_stops()
        self.log("Paso 3/3: Escanear oportunidades (find_setups)")
        self.scan_opportunities()
        self.snapshot()
        self.log("Ciclo completado")

    def run(self, interval_seconds=3600):
        self.running = True
        self.log(f"Bot v13 iniciado. Intervalo: {interval_seconds}s. Ctrl-C para parar.")
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

    def reset(self):
        self.log("RESET completo -> $500 cash, 0 posiciones, 0 cooldowns")
        self.state = {"version": "v13_event_driven",
                      "positions": {}, "cash": 500.0,
                      "history": [], "daily_snapshots": [],
                      "cooldowns": {},
                      "total_value": 500.0,
                      "last_update": self._today().isoformat()}
        self._save_state()


if __name__ == "__main__":
    bot = TradingBotV13()
    bot.connect_brokers()
    if "--once" in sys.argv:
        bot.run_cycle()
    elif "--scan" in sys.argv:
        bot.update_positions_prices()
        bot.scan_opportunities()
        bot.snapshot()
    elif "--reset" in sys.argv:
        bot.reset()
    elif "--sell-all" in sys.argv:
        bot.update_positions_prices()
        positions = bot.state.get("positions", {})
        for sym in list(positions.keys()):
            pos = positions[sym]
            price = pos.get("current_price", 0)
            qty = pos.get("qty", 0)
            entry = pos.get("entry_price", 0)
            pnl = (price - entry) * qty
            bot.state["cash"] += qty * price
            bot.log(f"SELL_ALL {sym} @ ${price:.2f} P&L ${pnl:+.2f}")
            bot.state["history"].append({
                "date": bot._today().isoformat(),
                "action": "SELL",
                "symbol": sym,
                "price": price,
                "qty": qty,
                "pnl": pnl,
                "reason": "MANUAL",
            })
            del positions[sym]
        bot._save_state()
        bot.snapshot()
    else:
        bot.run(interval_seconds=6 * 3600)
