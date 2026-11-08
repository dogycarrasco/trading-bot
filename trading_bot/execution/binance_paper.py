"""
Ejecutor paper para Binance.

Modos:
  - testnet=True  ->  https://testnet.binance.vision  (saldo ficticio)
  - testnet=False ->  https://api.binance.com         (LIVE: requiere API key real)

Para paper trading NUNCA usar testnet=False.

Setup:
    export BINANCE_API_KEY='tu_key_de_testnet'
    export BINANCE_API_SECRET='tu_secret_de_testnet'
"""
import os
from execution.base_broker import BaseBroker, Order, Position


class BinancePaperBroker(BaseBroker):
    name = "binance_paper"

    def __init__(self, testnet=True):
        self.testnet = testnet
        self.api_key = os.environ.get("BINANCE_API_KEY", "")
        self.api_secret = os.environ.get("BINANCE_API_SECRET", "")
        self.client = None

    def connect(self):
        try:
            from binance.client import Client
        except ImportError:
            print("ERROR: pip install python-binance")
            return False
        if not self.api_key or not self.api_secret:
            print("ERROR: configura BINANCE_API_KEY y BINANCE_API_SECRET")
            return False
        self.client = Client(self.api_key, self.api_secret, testnet=self.testnet)
        try:
            self.client.get_account_status()
            url = "testnet.binance.vision" if self.testnet else "api.binance.com (LIVE!)"
            print(f"Binance conectado ({url})")
            return True
        except Exception as e:
            print(f"Error conectando a Binance: {e}")
            return False

    def get_account_balance(self):
        if not self.client:
            return {}
        balances = self.client.get_account()["balances"]
        return {b["asset"]: float(b["free"]) for b in balances if float(b["free"]) > 0}

    def place_order(self, order):
        if not self.client:
            return {"status": "error", "msg": "not connected"}
        try:
            result = self.client.create_order(
                symbol=order.symbol,
                side=order.side,
                type=order.order_type,
                quantity=order.qty,
                price=order.limit_price,
                stopPrice=order.stop_price,
            )
            return {
                "status": "filled" if result.get("status") == "FILLED" else result.get("status", "unknown"),
                "order_id": result.get("orderId"),
                "raw": result,
            }
        except Exception as e:
            return {"status": "error", "msg": str(e)}

    def cancel_order(self, order_id):
        return False

    def get_position(self, symbol):
        if not self.client:
            return None
        base = symbol.replace("USDT", "").replace("BUSD", "")
        try:
            bal = self.client.get_asset_balance(base)
            if not bal or float(bal["free"]) <= 0:
                return None
            qty = float(bal["free"])
            ticker = self.client.get_symbol_ticker(symbol=symbol)
            price = float(ticker["price"])
            return Position(symbol=symbol, qty=qty, side="LONG", entry_price=0.0, current_price=price, pnl=0.0)
        except Exception:
            return None

    def get_open_orders(self):
        if not self.client:
            return []
        try:
            return self.client.get_open_orders()
        except Exception:
            return []