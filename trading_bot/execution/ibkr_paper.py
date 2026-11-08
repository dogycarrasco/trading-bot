"""
Ejecutor paper para Interactive Brokers (IBKR).

IBKR requiere TWS (Trader Workstation) o IB Gateway abierto en la maquina donde corre el bot.

Para $500 IBKR NO es viable para operar (minimos de posicion). Se recomienda
usar IBKR solo para datos y ejecutar en Binance/Alpaca.

Dependencias:
    pip install ib_insync

Puertos:
    7497 = paper trading (TWS)
    7496 = live trading (TWS)

Setup TWS:
    1. Abrir TWS
    2. Configure -> Settings -> API -> Settings
    3. Activar "Enable ActiveX and Socket Clients"
    4. Desactivar "Read-Only API"
    5. Trusted IPs: anadir 127.0.0.1
    6. Socket port: 7497 (paper)
"""
from execution.base_broker import BaseBroker, Order, Position


class IBKRPaperBroker(BaseBroker):
    name = "ibkr_paper"

    def __init__(self, host="127.0.0.1", port=7497, client_id=1):
        self.host = host
        self.port = port
        self.client_id = client_id
        self.ib = None

    def connect(self):
        try:
            from ib_insync import IB
        except ImportError:
            print("ERROR: pip install ib_insync")
            return False
        try:
            self.ib = IB()
            self.ib.connect(self.host, self.port, clientId=self.client_id)
            print(f"IBKR conectado ({self.host}:{self.port}, paper trading)")
            return True
        except Exception as e:
            print(f"Error conectando a IBKR: {e}")
            print("Asegurate de tener TWS abierto con API activado en puerto 7497 (paper)")
            return False

    def get_account_balance(self):
        if not self.ib:
            return {}
        try:
            acct = self.ib.accountSummary()
            out = {}
            for a in acct:
                out[a.tag] = float(a.value) if a.value.replace('.', '').replace('-', '').isdigit() else a.value
            return out
        except Exception:
            return {}

    def place_order(self, order):
        if not self.ib:
            return {"status": "error", "msg": "not connected"}
        try:
            from ib_insync import Stock, MarketOrder, LimitOrder, StopOrder
            contract = Stock(order.symbol, 'SMART', 'USD')
            self.ib.qualifyContracts(contract)
            if order.order_type.upper() == 'MARKET':
                ib_order = MarketOrder(order.side.upper(), order.qty)
            elif order.order_type.upper() == 'LIMIT':
                ib_order = LimitOrder(order.side.upper(), order.qty, order.limit_price)
            else:
                return {"status": "error", "msg": f"order type {order.order_type} not supported"}
            trade = self.ib.placeOrder(contract, ib_order)
            return {"status": "submitted", "order_id": trade.order.orderId, "raw": str(trade)}
        except Exception as e:
            return {"status": "error", "msg": str(e)}

    def cancel_order(self, order_id):
        if not self.ib:
            return False
        try:
            trades = self.ib.openTrades()
            for t in trades:
                if str(t.order.orderId) == str(order_id):
                    self.ib.cancelOrder(t.order)
                    return True
            return False
        except Exception:
            return False

    def get_position(self, symbol):
        if not self.ib:
            return None
        try:
            positions = self.ib.positions()
            for p in positions:
                if p.contract.symbol == symbol:
                    return Position(
                        symbol=symbol,
                        qty=float(p.position),
                        side="LONG" if p.position > 0 else "SHORT",
                        entry_price=float(p.avgCost),
                        current_price=0.0,
                        pnl=0.0,
                    )
            return None
        except Exception:
            return None

    def get_open_orders(self):
        if not self.ib:
            return []
        try:
            return [str(t) for t in self.ib.openTrades()]
        except Exception:
            return []