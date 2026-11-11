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
    print("Descargando datos...")
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


class TradingBot:

    def __init__(self):
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
        self.breakout_lookback = 20

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
        return {"positions": {}, "cash": 500.0, "history": [], "trade_history": []}

    def _save_state(self):
        try:
            with open(self.state_path, "w") as f:
                json.dump(self.state, f, indent=2, default=str)
        except Exception as e:
            self.log(f"ERROR guardando: {e}")

    def has_positions(self):
        return len(self.state.get("positions", {})) > 0

    def get_total_value(self):
        cash = self.state.get("cash", 0)
        positions = self.state.get("positions", {})
        invested = sum(p.get("value", 0) for p in positions.values())
        return cash + invested

    def find_breakout_opportunities(self):
        """Encuentra activos que rompieron su maximo de 20 dias."""
        breakouts = []
        for sym, exch, desc in UNIVERSE_V8:
            df = load_tv_data(sym, exch)
            if df.empty or len(df) < self.breakout_lookback + 1:
                continue
            close = df["close"]
            high_20 = df["high"].iloc[-self.breakout_lookback:-1].max()
            current = float(close.iloc[-1])
            volume_avg = df["volume"].iloc[-20:].mean()
            current_vol = float(df["volume"].iloc[-1])
            vol_ratio = current_vol / volume_avg if volume_avg > 0 else 1
            if current > high_20 and vol_ratio > 1.2:
                pct_above = ((current / high_20) - 1) * 100
                breakouts.append({"symbol": sym, "price": current, "pct_above": pct_above, "vol_ratio": vol_ratio})
        return breakouts

    def buy_asset(self, symbol, amount):
        """Compra un activo con la cantidad indicada."""
        df = load_tv_data(symbol, "")
        if df.empty:
            return False
        price = float(df["close"].iloc[-1])
        if price <= 0:
            return False
        qty = amount / price
        positions = self.state.get("positions", {})
        positions[symbol] = {
            "qty": qty,
            "entry_price": price,
            "value": amount,
            "weight": 1.0,
            "peak_price": price,
        }
        self.state["positions"] = positions
        self.state["cash"] = self.state.get("cash", 0) - amount
        self.log(f"  >>> COMPRA {symbol}: qty={qty:.6f} @ ${price:.2f} = ${amount:.2f}")
        return True

    def sell_asset(self, symbol, reason):
        """Vende un activo."""
        positions = self.state.get("positions", {})
        if symbol not in positions:
            return
        pos = positions[symbol]
        df = load_tv_data(symbol, "")
        if df.empty:
            return
        price = float(df["close"].iloc[-1])
        qty = pos.get("qty", 0)
        value = qty * price
        pnl = (price - pos.get("entry_price", price)) * qty
        self.state["cash"] = self.state.get("cash", 0) + value
        del positions[symbol]
        self.state["positions"] = positions
        self.log(f"  >>> VENTA {symbol} @ ${price:.2f} (P&L {pnl:+.2f}) - {reason}")
        trade = {
            "symbol": symbol,
            "entry_price": pos.get("entry_price", 0),
            "exit_price": price,
            "qty": qty,
            "pnl": pnl,
            "exit_reason": reason,
            "date": datetime.now(timezone.utc).isoformat(),
        }
        self.state.setdefault("trade_history", []).append(trade)

    def check_stops(self):
        """Verifica stop-loss y trailing stop para todas las posiciones."""
        positions = self.state.get("positions", {})
        to_sell = []
        for asset, pos in list(positions.items()):
            df = load_tv_data(asset, "")
            if df.empty:
                continue
            cur_price = float(df["close"].iloc[-1])
            entry = pos.get("entry_price", cur_price)
            peak = pos.get("peak_price", entry)
            pnl_pct = ((cur_price / entry) - 1) * 100 if entry > 0 else 0
            if cur_price > peak:
                peak = cur_price
                pos["peak_price"] = peak
            if pnl_pct / 100 <= self.stop_loss_pct:
                to_sell.append((asset, "STOP_LOSS"))
            elif gain >= self.trailing_trigger and drop <= self.trailing_drop:
                to_sell.append((asset, "TRAILING_STOP"))
            gain = (peak / entry - 1) if entry > 0 else 0
            drop = (cur_price / peak - 1) if peak > 0 else 0
            if gain >= self.trailing_trigger and drop <= self.trailing_drop:
                if asset not in [s[0] for s in to_sell]:
                    to_sell.append((asset, "TRAILING_STOP"))
        for asset, reason in to_sell:
            self.sell_asset(asset, reason)
        return len(to_sell) > 0

    def update_prices(self):
        """Actualiza precios y P&L de las posiciones."""
        positions = self.state.get("positions", {})
        updated = {}
        total_value = self.state.get("cash", 0)
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
                updated[asset] = {
                    "qty": qty,
                    "entry_price": entry,
                    "value": value,
                    "current_price": cur_price,
                    "weight": pos.get("weight", 1),
                    "peak_price": peak,
                }
                self.log(f"  {asset}: ${cur_price:.2f} x {qty:.6f} = ${value:.2f} (P&L {pnl:+.2f}, {pnl_pct:+.2f}%)")
            else:
                updated[asset] = pos
                total_value += pos.get("value", 0)
        self.state["positions"] = updated
        return total_value

    def run_cycle(self):
        download_universe_data()
        self.log("=" * 60)
        self.log("Ciclo bot v13 (event-driven)")
        self.log("=" * 60)

        # 1. Actualizar precios
        total_value = self.update_prices()
        if not self.has_positions():
            self.log("Sin posiciones. Total: $500.00 (cash)")

        # 2. Verificar stops (vender si toca)
        sold = self.check_stops()
        if sold:
            total_value = self.get_total_value()
            self.log(f"  Total tras ventas: ${total_value:.2f}")

        # 3. Market score + ML
        ml_pred = 0
        ml_acc = 0
        try:
            features = load_macro_features()
        except Exception:
            features = pd.DataFrame()

        try:
            spy_df = load_tv_data("SPY", "AMEX")
            spy_feat = self.ml_predictor.prepare_features(spy_df, features)
            ml_pred, ml_acc = self.ml_predictor.train_predict(spy_feat)
            self.log(f"ML: pred={ml_pred} acc={ml_acc:.1%}")
        except Exception as e:
            self.log(f"ML error: {e}")

        vix = 25
        fg = 50
        ff = 4.5
        yinv = False
        spy_above = True
        spy_price = 500.0
        spy_mom = 0
        try:
            if not features.empty:
                lf = features.iloc[-1]
                vix = lf.get("vix", 25)
                fg = lf.get("fear_greed", 50)
                ff = lf.get("fed_funds", 4.5)
                yinv = bool(lf.get("yield_inverted", 0))
            if not spy_df.empty:
                spy_price = float(spy_df["close"].iloc[-1])
                ema200 = spy_df["close"].rolling(200).mean().iloc[-1]
                spy_above = spy_price > float(ema200) if not pd.isna(ema200) else True
                spy_mom = float(spy_df["close"].pct_change(20).iloc[-1] * 100) if len(spy_df) > 20 else 0
        except Exception:
            pass

        score = compute_market_score(vix, fg, spy_above, yinv, ff, spy_mom, spy_mom * 2)
        regime = score_to_regime(score)
        eq_pct, cash_pct, hedge_pct = score_to_allocation(score)
        self.log(f"Market Score: {score:+.0f} ({regime}) -> {eq_pct*100:.0f}% eq / {cash_pct*100:.0f}% cash / {hedge_pct*100:.0f}% hedge")

        # 4. HEDGE
        try:
            if score < -20 and len(self.hedge_manager.positions) == 0:
                hb = total_value * hedge_pct
                pp = black_scholes_put(spy_price, spy_price * 0.95, 30/365, 0.045, vix / 100)
                if pp > 0:
                    nc = max(1, int(hb / (pp * 100)))
                    self.hedge_manager.buy_put(spy_price, vix, quantity=nc)
                    self.log(f"  HEDGE: {nc} puts SPY @ ${pp:.2f}")
            if self.hedge_manager.positions:
                self.hedge_manager.update_positions(spy_price, vix)
                hs = self.hedge_manager.get_summary()
                self.log(f"  HEDGE: {hs['open_positions']} puts, P&L ${hs['pnl']:.2f}")
            if score > 0 and self.hedge_manager.positions:
                self.hedge_manager.close_all(spy_price, vix)
                self.log("  HEDGE: cerrados")
            self.state["market_score"] = score
            self.state["market_regime"] = regime
            self.state["hedge_summary"] = self.hedge_manager.get_summary()
        except Exception as e:
            self.log(f"  HEDGE error: {e}")

        # 5. ENTRADAS: solo comprar si hay cash Y hay senales
        cash = self.state.get("cash", 0)
        if cash > 50 and (ml_pred == 1 or score >= 20):
            self.log(f"Buscando oportunidades (cash=${cash:.2f})...")
            breakouts = self.find_breakout_opportunities()
            if breakouts:
                self.log(f"  Breakouts detectados: {len(breakouts)}")
                for b in breakouts[:3]:
                    self.log(f"    {b['symbol']}: ${b['price']:.2f} (+{b['pct_above']:.1f}%, vol x{b['vol_ratio']:.1f})")
                n_to_buy = min(3, len(breakouts))
                amount_each = cash / n_to_buy
                for b in breakouts[:n_to_buy]:
                    if not self.has_positions() or len(self.state.get("positions", {})) < 5:
                        self.buy_asset(b["symbol"], amount_each)
            elif not self.has_positions():
                self.log("  Sin breakouts. Sin posiciones. Esperando oportunidad...")
        elif not self.has_positions() and cash > 50:
            self.log(f"Sin senal de compra (ML={ml_pred}, score={score}). Esperando...")

        # 6. P&L total
        total_value = self.get_total_value()
        total_pnl = total_value - 500
        total_pnl_pct = (total_pnl / 500) * 100
        self.log(f"Total: ${total_value:.2f} (P&L {total_pnl:+.2f}, {total_pnl_pct:+.2f}%)")

        snapshot = {"date": datetime.now(timezone.utc).isoformat(), "total_value": total_value, "cash": self.state.get("cash", 0), "pnl": total_pnl, "pnl_pct": total_pnl_pct, "ml_pred": ml_pred, "market_score": score}
        self.state.setdefault("daily_snapshots", []).append(snapshot)
        if len(self.state["daily_snapshots"]) > 90:
            self.state["daily_snapshots"] = self.state["daily_snapshots"][-90:]
        self._save_state()
        self.log("Ciclo completado")


if __name__ == "__main__":
    bot = TradingBot()
    if "--once" in sys.argv:
        bot.run_cycle()
    elif "--rebalance" in sys.argv:
        bot.run_cycle()
    else:
        bot.run_cycle()
