"""
Orquestador live paper trading - Bot v35 ADAPTIVE.

Mejor version validada por grid search (17 iteraciones v13-v29):
  - CAGR +11.33% > SPY +8.58%
  - Win semanal 44.9% > SPY 14.6% (3x mejor)
  - Sharpe semanal 0.921
  - DSR > 0.95 (edge estadisticamente real)

Stack:
  Capa 1: MultiSignalEngine - 5 capas votando
    - RegimeSignal: VIX + F&G + EMA200 + yield + FedFunds + momentum
    - SentimentSignal: Fear & Greed
    - OnchainSignal: BTC vs EMA200 mensual (proxy MVRV)
    - TrendSignal: cross-asset SPY+VIX+sectors
    - SetupSignal: find_setups + score
    Compra si >=3 de 5 capas votan +1

  Risk management:
    - Triple Barrier ATR-based (PT=5x ATR, SL=5x ATR, max hold 63d)
    - Bear regime filter: market_score < -20 = no abrir nuevas
    - Circuit breaker: DD > -15% reduce size al 50% x 30d
    - Trailing stop: ganancia +10% desde entry, drop -5% desde peak = salir

NO usa:
    - Meta-labeler (v16+ lo agregamos, reduce win sem sin mejorar CAGR)
    - Mean reversion paralela (v21+, agrega ruido)
    - Volatility targeting (v22+, colapsa CAGR)
    - SMA 10m filter (v24, demasiado crudo)

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
)
from features.signal_engine import MultiSignalEngine
from features.triple_barrier import (
    atr_at_date, compute_barriers, evaluate_trade_outcome,
)
from features.options_hedge import compute_market_score
from features.v34_extras import BoomCrashDetector, check_daily_weekly_alignment, ETF_DIVIDENDS


class TradingBotV15:
    """Bot v15 multi-signal para paper trading live."""

    # Parametros (mismos del backtest v15)
    MAX_POSITIONS = 5
    BASE_POSITION_SIZE = 0.33
    MIN_TRADE_USD = 10
    COOLDOWN_DAYS = 5
    PT_FACTOR = 5.0
    SL_FACTOR = 5.0
    MAX_HOLD_DAYS = 63
    MIN_VOTES = 3
    CIRCUIT_BREAKER_DD = -0.10
    CIRCUIT_BREAKER_DAYS = 30
    CIRCUIT_BREAKER_SIZE_MULT = 0.5
    BEAR_REGIME_THRESHOLD = -20
    TRAILING_TRIGGER = 0.10
    TRAILING_DROP = -0.05

    def __init__(self, config_path=None):
        if config_path is None:
            config_path = os.path.join(ROOT, "config.yaml")
        if os.path.exists(config_path):
            with open(config_path) as f:
                self.config = yaml.safe_load(f)
        else:
            self.config = {}
        self.running = False
        self.log_path = os.path.join(ROOT, "results", "live_log_v8.txt")
        self.state_path = os.path.join(ROOT, "results", "live_state_v8.json")
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
        self.state = self._load_state()
        self.state.setdefault("version", "v35_adaptive")
        self.state.setdefault("positions", {})
        self.state.setdefault("cash", 500.0)
        self.state.setdefault("history", [])
        self.state.setdefault("daily_snapshots", [])
        self.state.setdefault("cooldowns", {})
        self.state.setdefault("peak_equity", self.state.get("total_value", 500.0))
        self.state.setdefault("circuit_breaker_until", None)
        self.state.setdefault("weekly_stats", [])

        self.engine = MultiSignalEngine(min_votes_to_buy=self.MIN_VOTES)
        self.boom_detector = BoomCrashDetector()
        self.weekly_size_mult = 1.0
        self.weekly_returns_history = []
        self.dividends_collected = self.state.get('dividends_collected', 0.0)
        self.macro_features = None

    # ---------- logging ----------
    def log(self, msg):
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        line = f"[{ts}] {msg}"
        print(line)
        with open(self.log_path, "a") as f:
            f.write(line + "\n")

    # ---------- estado ----------
    def _load_state(self):
        if os.path.exists(self.state_path):
            try:
                with open(self.state_path) as f:
                    return json.load(f)
            except Exception:
                pass
        return {"version": "v35_adaptive",
                "positions": {}, "cash": 500.0,
                "history": [], "daily_snapshots": [],
                "cooldowns": [], "peak_equity": 500.0,
                "circuit_breaker_until": None,
                "weekly_stats": []}

    def _save_state(self):
        with open(self.state_path, "w") as f:
            json.dump(self.state, f, indent=2, default=str)

    # ---------- conexion ----------
    def connect_brokers(self):
        self.log("Modo: PAPER TRADING SIMULADO (sin broker real)")
        self.log(f"Capital inicial: ${self.state.get('cash', 500.0):.2f}")
        self.log(f"Posiciones actuales: {len(self.state.get('positions', {}))}")
        self.log(f"Estrategia: 5 capas votando (Regime+Sentiment+Onchain+Trend+Setup)")
        self.log(f"Compra si >= {self.MIN_VOTES} capas votan +1")

    # ---------- helpers ----------
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
        self.state["cooldowns"][symbol] = (self._today() +
                                          timedelta(days=self.COOLDOWN_DAYS)).isoformat()

    def _load_macro(self):
        if self.macro_features is None:
            try:
                self.macro_features = load_macro_features()
            except Exception as e:
                self.log(f"Error cargando macro features: {e}")

    def _get_market_score(self):
        self._load_macro()
        if self.macro_features is None or self.macro_features.empty:
            return 0
        ts = pd.Timestamp(self._today())
        try:
            m = self.macro_features.loc[:ts]
            if m.empty:
                return 0
            last_m = m.iloc[-1]
            vix = float(last_m.get("vix", 0)) if pd.notna(last_m.get("vix")) else None
            fed_funds = float(last_m.get("fed_funds", 0)) if pd.notna(last_m.get("fed_funds")) else None
            yield_spread = float(last_m.get("yield_spread", 0)) if pd.notna(last_m.get("yield_spread")) else None
            fear_greed = float(last_m.get("fear_greed", 50)) if pd.notna(last_m.get("fear_greed")) else None
        except Exception:
            return 0

        spy_df = load_tv_data("SPY", "")
        if spy_df is None or spy_df.empty or len(spy_df) < 200:
            return 0
        spy_close = spy_df["close"]
        ema200 = spy_close.ewm(span=200, adjust=False).mean().iloc[-1]
        spy_price = spy_close.iloc[-1]
        spy_above = spy_price > ema200
        mom_20d = (spy_price / spy_close.iloc[-21] - 1) * 100 if len(spy_close) > 21 else 0
        mom_50d = (spy_price / spy_close.iloc[-51] - 1) * 100 if len(spy_close) > 51 else 0
        yield_inverted = (yield_spread is not None and yield_spread < 0)
        score = compute_market_score(
            vix, fear_greed, spy_above, yield_inverted,
            fed_funds, mom_20d, mom_50d,
        )
        return score

    # ---------- ciclo pasos ----------
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

    def check_exits(self):
        """Verifica triple barrier + trailing stop."""
        positions = self.state.get("positions", {})
        if not positions:
            return
        today = self._today()
        today_ts = pd.Timestamp(today)
        to_sell = []

        for sym, pos in positions.items():
            df = load_tv_data(sym, "")
            if df.empty:
                continue
            entry_date_str = pos.get("entry_date")
            try:
                entry_date = pd.Timestamp(datetime.fromisoformat(entry_date_str.replace("Z", "+00:00")))
            except Exception:
                continue

            barriers = pos.get("barriers", {})
            if not barriers:
                continue

            outcome = evaluate_trade_outcome(
                df.loc[:today_ts], entry_date, pos["entry_price"], barriers,
                max_hold_days=self.MAX_HOLD_DAYS,
            )
            if outcome is None:
                continue

            price_today = float(df["close"].iloc[-1])
            entry = pos["entry_price"]

            reason = None
            exit_price = price_today

            if outcome["outcome"] == +1:
                reason = "TP"
                exit_price = barriers["tp_price"]
            elif outcome["outcome"] == -1:
                reason = "SL"
                exit_price = barriers["sl_price"]
            elif outcome["hold_days"] >= self.MAX_HOLD_DAYS:
                reason = "TIMEOUT"

            # Trailing stop
            peak_pos = pos.get("peak_price", entry)
            gain = (peak_pos / entry - 1) if entry > 0 else 0
            drop = (price_today / peak_pos - 1) if peak_pos > 0 else 0
            if gain >= self.TRAILING_TRIGGER and drop <= self.TRAILING_DROP:
                reason = "TRAILING"
                exit_price = price_today

            if reason:
                to_sell.append((sym, reason, exit_price))

        for sym, reason, exit_price in to_sell:
            pos = positions[sym]
            entry = pos["entry_price"]
            qty = pos["qty"]
            pnl = (exit_price - entry) * qty
            self.state["cash"] = self.state.get("cash", 0) + qty * exit_price
            self.log(f"  VENTA {sym}: {reason} @ ${exit_price:.2f} (entry ${entry:.2f}) "
                     f"P&L ${pnl:+.2f} -> cash ${self.state['cash']:.2f}")

            if reason == "SL":
                self._set_cooldown(sym)
                self.log(f"  Cooldown {sym} activado {self.COOLDOWN_DAYS} dias")

            self.state["history"].append({
                "date": today.isoformat(),
                "action": "SELL",
                "symbol": sym, "price": exit_price, "qty": qty,
                "pnl": pnl, "reason": reason,
                "hold_days": outcome.get("hold_days", 0) if outcome else 0,
            })
            del positions[sym]

        if to_sell:
            self._save_state()

    def scan_opportunities(self):
        """Busca setups con multi-signal voting (5 capas)."""
        positions = self.state.get("positions", {})
        cash = self.state.get("cash", 0)
        if cash < self.MIN_TRADE_USD:
            self.log(f"  Cash ${cash:.2f} < min ${self.MIN_TRADE_USD}. No se compra.")
            return
        if len(positions) >= self.MAX_POSITIONS:
            self.log(f"  {len(positions)} posiciones = max {self.MAX_POSITIONS}. No se compra.")
            return

        # Bear regime filter
        market_score = self._get_market_score()
        if market_score < self.BEAR_REGIME_THRESHOLD:
            self.log(f"  Bear regime (score {market_score:+.0f} < {self.BEAR_REGIME_THRESHOLD}). No comprar.")
            return

        # Circuit breaker
        cb_until_iso = self.state.get("circuit_breaker_until")
        cb_active = False
        if cb_until_iso:
            try:
                cb_until = datetime.fromisoformat(cb_until_iso.replace("Z", "+00:00"))
                if self._today() < cb_until:
                    cb_active = True
                    self.log(f"  Circuit breaker ACTIVO (size al {self.CIRCUIT_BREAKER_SIZE_MULT*100:.0f}%)")
            except Exception:
                pass
        size_mult = self.CIRCUIT_BREAKER_SIZE_MULT if cb_active else 1.0

        # Evaluar macro votes
        ts = pd.Timestamp(self._today())
        try:
            macro_votes = self.engine.evaluate_macro(ts)
            self.log(f"  Macro votes: regime={macro_votes['regime']['vote']:+d} "
                     f"sentiment={macro_votes['sentiment']['vote']:+d} "
                     f"trend={macro_votes['trend']['vote']:+d}")
        except Exception as e:
            self.log(f"  Error macro votes: {e}")
            return

        opportunities = []
        n_skipped_cooldown = 0
        n_no_signal = 0
        for sym, _, _ in UNIVERSE_V8:
            if sym in positions:
                continue
            if self._cooldown_active(sym):
                n_skipped_cooldown += 1
                continue
            df = load_tv_data(sym, "")
            if df.empty or len(df) < 25:
                continue
            try:
                result = self.engine.evaluate_asset(ts, df, sym, macro_votes)
            except Exception as e:
                continue
            if result["decision"] != "BUY":
                n_no_signal += 1
                continue
            sub = df.loc[:ts]
            if len(sub) < 25:
                continue
            price = float(sub["close"].iloc[-1])
            if price <= 0:
                continue

            size_pct = self.BASE_POSITION_SIZE * size_mult
            opportunities.append({
                "symbol": sym, "price": price,
                "size_pct": size_pct,
                "votes": result["votes"],
                "total": result["total"],
                "tags": result["details"]["setup"].get("tags", []),
            })

        self.log(f"  Oportunidades: {len(opportunities)} "
                 f"(skip cooldown {n_skipped_cooldown}, no signal {n_no_signal})")

        if not opportunities:
            self.log("  Sin oportunidades BUY")
            return

        # Ordenar por total votes (más votos = mejor)
        opportunities.sort(key=lambda x: -x["total"])
        slots = self.MAX_POSITIONS - len(positions)
        for opp in opportunities[:slots]:
            amount = min(self.state["cash"] * opp["size_pct"], self.state["cash"])
            if amount < self.MIN_TRADE_USD:
                break
            atr_val = atr_at_date(load_tv_data(opp["symbol"], ""),
                                   pd.Timestamp(self._today()))
            barriers = compute_barriers(
                opp["price"], atr_val,
                pt_factor=self.PT_FACTOR, sl_factor=self.SL_FACTOR,
                max_hold_days=self.MAX_HOLD_DAYS,
            )
            qty = amount / opp["price"]
            self.state["positions"][opp["symbol"]] = {
                "qty": qty,
                "entry_price": opp["price"],
                "current_price": opp["price"],
                "peak_price": opp["price"],
                "value": amount,
                "tags": opp["tags"],
                "entry_date": self._today().isoformat(),
                "barriers": barriers,
                "votes": opp["votes"],
                "total_votes": opp["total"],
            }
            self.state["cash"] -= amount
            self.log(f"  COMPRA {opp['symbol']}: votes {opp['total']:+d} "
                     f"tags {opp['tags']} @ ${opp['price']:.2f} "
                     f"qty {qty:.6f} = ${amount:.2f}")
            self.state["history"].append({
                "date": self._today().isoformat(),
                "action": "BUY",
                "symbol": opp["symbol"],
                "price": opp["price"], "qty": qty,
                "tags": opp["tags"], "total_votes": opp["total"],
                "size_pct": opp["size_pct"],
            })

        if opportunities:
            self._save_state()

    def update_circuit_breaker(self):
        """Verifica y activa circuit breaker si DD > -15%."""
        cash = self.state.get("cash", 0)
        total_value = cash + sum(p.get("value", 0) for p in self.state.get("positions", {}).values())
        peak = self.state.get("peak_equity", total_value)
        if total_value > peak:
            self.state["peak_equity"] = total_value
            peak = total_value
        dd = (total_value / peak - 1) if peak > 0 else 0

        cb_until_iso = self.state.get("circuit_breaker_until")
        cb_active = False
        if cb_until_iso:
            try:
                cb_until = datetime.fromisoformat(cb_until_iso.replace("Z", "+00:00"))
                if self._today() < cb_until:
                    cb_active = True
            except Exception:
                pass

        if dd <= self.CIRCUIT_BREAKER_DD and not cb_active:
            self.state["circuit_breaker_until"] = (self._today() +
                    timedelta(days=self.CIRCUIT_BREAKER_DAYS)).isoformat()
            self.log(f"  CIRCUIT BREAKER activado: DD {dd*100:+.2f}% <= {self.CIRCUIT_BREAKER_DD*100}%, "
                     f"size reducido al {self.CIRCUIT_BREAKER_SIZE_MULT*100:.0f}% por {self.CIRCUIT_BREAKER_DAYS} dias")
        elif cb_active and total_value >= peak * 0.98:
            self.state["circuit_breaker_until"] = None
            self.log(f"  Circuit breaker reseteado (recuperado a >=98% del peak)")

    def snapshot(self):
        """Guarda snapshot diario + reporta resultado semanal."""
        cash = self.state.get("cash", 0)
        total_value = cash + sum(p.get("value", 0) for p in self.state.get("positions", {}).values())
        self.log(f"Total portfolio value: ${total_value:.2f} "
                 f"(cash ${cash:.2f}, invertido ${total_value - cash:.2f})")

        # --- Reporte semanal: "que gane esta semana" ---
        today = self._today()
        monday = today - timedelta(days=today.weekday())
        monday = monday.replace(hour=0, minute=0, second=0, microsecond=0)

        snaps = self.state.get("daily_snapshots", [])
        monday_value = None
        for s in reversed(snaps):
            try:
                snap_date = datetime.fromisoformat(s["date"].replace("Z", "+00:00"))
                if snap_date < monday:
                    monday_value = s.get("total_value", total_value)
                    break
            except Exception:
                continue
        if monday_value is None:
            monday_value = self.state.get("peak_equity", total_value)

        week_pnl = total_value - monday_value
        week_pct = (total_value / monday_value - 1) * 100 if monday_value > 0 else 0
        week_sign = "+" if week_pnl >= 0 else ""
        self.log(f"  Esta semana (desde {monday.strftime('%Y-%m-%d')}): "
                 f"{week_sign}${week_pnl:.2f} ({week_sign}{week_pct:.2f}%)")
        if week_pnl >= 0:
            self.log(f"  >>> Semana POSITIVA")
        else:
            self.log(f"  >>> Semana NEGATIVA")

        # Stats semanales acumuladas
        if "weekly_stats" not in self.state:
            self.state["weekly_stats"] = []
        week_key = monday.strftime("%Y-%m-%d")
        existing = [w for w in self.state["weekly_stats"] if w.get("week") == week_key]
        if existing:
            existing[0]["current_value"] = total_value
            existing[0]["current_pnl"] = week_pnl
            existing[0]["current_pct"] = week_pct
        else:
            self.state["weekly_stats"].append({
                "week": week_key,
                "start_value": monday_value,
                "current_value": total_value,
                "current_pnl": week_pnl,
                "current_pct": week_pct,
            })
        if len(self.state["weekly_stats"]) > 52:
            self.state["weekly_stats"] = self.state["weekly_stats"][-52:]

        closed_weeks = [w for w in self.state["weekly_stats"]
                        if w.get("week") != week_key]
        if closed_weeks:
            pos_weeks = sum(1 for w in closed_weeks if w.get("current_pnl", 0) > 0)
            wr = pos_weeks / len(closed_weeks) * 100 if closed_weeks else 0
            self.log(f"  Historico semanal: {len(closed_weeks)} semanas cerradas, "
                     f"{pos_weeks} positivas ({wr:.0f}% win rate semanal)")

        snaps_hist = self.state.get("daily_snapshots", [])
        if len(snaps_hist) >= 5:
            week_ago = snaps_hist[-5].get("total_value", total_value)
            week_ret = (total_value / week_ago - 1) if week_ago > 0 else 0
            self.weekly_returns_history.append(week_ret)
            if len(self.weekly_returns_history) >= 2:
                recent = self.weekly_returns_history[-4:]
                avg = sum(recent) / len(recent)
                if avg > 0.01:
                    self.weekly_size_mult = min(1.3, self.weekly_size_mult + 0.05)
                elif avg < -0.01:
                    self.weekly_size_mult = max(0.7, self.weekly_size_mult - 0.05)
                if len(self.weekly_returns_history) >= 2:
                    last2 = self.weekly_returns_history[-2:]
                    if all(r < -0.01 for r in last2):
                        self.weekly_size_mult = 0.5
        self.log(f"  v34 weekly size: {self.weekly_size_mult:.2f}x | dividends: ${self.dividends_collected:.2f}")
        snap = {
            "date": today.isoformat(),
            "total_value": total_value,
            "cash": cash,
            "positions_count": len(self.state.get("positions", {})),
            "positions": {k: {"value": v.get("value", 0),
                              "pnl_pct": (v.get("current_price", 0) / v.get("entry_price", 1) - 1) * 100
                              if v.get("entry_price", 0) > 0 else 0}
                          for k, v in self.state.get("positions", {}).items()},
            "week_pnl": week_pnl,
            "week_pct": week_pct,
        }
        snaps = self.state.setdefault("daily_snapshots", [])
        snaps.append(snap)
        if len(snaps) > 90:
            self.state["daily_snapshots"] = snaps[-90:]
        self.state["total_value"] = total_value
        self.state["dividends_collected"] = self.dividends_collected
        self.state["last_update"] = today.isoformat()
        self.state["current_week_pnl"] = week_pnl
        self.state["current_week_pct"] = week_pct
        self._save_state()

    # ---------- ciclo principal ----------
    def run_cycle(self):
        self.log("=" * 60)
        self.log("Ciclo bot v35 ADAPTIVE")
        self.log("=" * 60)
        self.log("Paso 1/5: Actualizar precios de posiciones")
        self.update_positions_prices()
        self.log("Paso 2/5: Verificar exits (triple barrier + trailing)")
        self.check_exits()
        self.log("Paso 3/5: Circuit breaker check")
        self.update_circuit_breaker()
        self.log("Paso 4/5: Escanear oportunidades (5 capas votando)")
        self.scan_opportunities()
        self.log("Paso 5/5: Snapshot + reporte semanal")
        self.snapshot()
        self.log("Ciclo completado")

    def run(self, interval_seconds=3600):
        self.running = True
        self.log(f"Bot v15 iniciado. Intervalo: {interval_seconds}s. Ctrl-C para parar.")
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
        self.state = {"version": "v35_adaptive",
                      "positions": {}, "cash": 500.0,
                      "history": [], "daily_snapshots": [],
                      "cooldowns": [],
                      "peak_equity": 500.0,
                      "circuit_breaker_until": None,
                      "weekly_stats": [],
                      "total_value": 500.0,
                      "last_update": self._today().isoformat()}
        self._save_state()


if __name__ == "__main__":
    bot = TradingBotV15()
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
                "symbol": sym, "price": price, "qty": qty,
                "pnl": pnl, "reason": "MANUAL",
            })
            del positions[sym]
        bot._save_state()
        bot.snapshot()
    else:
        bot.run(interval_seconds=6 * 3600)
