"""
Base broker interface - Fix 1.

binance_paper.py e ibkr_paper.py importan BaseBroker, Order, Position
de este modulo. Antes no existia -> ImportError.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Order:
    """Orden de trading."""
    symbol: str
    side: str                      # "buy" | "sell"
    order_type: str                # "market" | "limit" | "stop"
    qty: float
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    status: str = "pending"        # pending | filled | cancelled | rejected
    order_id: Optional[str] = None
    filled_price: Optional[float] = None
    filled_qty: float = 0.0


@dataclass
class Position:
    """Posicion abierta en el broker."""
    symbol: str
    qty: float
    side: str                      # "long" | "short"
    entry_price: float
    current_price: float
    pnl: float = 0.0
    unrealized_pnl: float = 0.0

    def __post_init__(self):
        if self.current_price and self.entry_price:
            diff = self.current_price - self.entry_price
            mult = 1.0 if self.side == "long" else -1.0
            self.unrealized_pnl = diff * self.qty * mult
            self.pnl = self.unrealized_pnl


class BaseBroker:
    """
    Interfaz base que todo broker (paper o live) debe implementar.

    Subclases: BinancePaperBroker, IBKRPaperBroker, AlpacaPaperBroker.
    """

    def __init__(self, account_id: str = "paper"):
        self.account_id = account_id
        self.orders: list[Order] = []
        self.positions: dict[str, Position] = {}

    def place_order(self, order: Order) -> Order:
        """Envia una orden. Debe actualizar order.status y devolverla."""
        raise NotImplementedError("Subclase debe implementar place_order")

    def get_position(self, symbol: str) -> Optional[Position]:
        """Devuelve la posicion abierta para symbol, o None."""
        raise NotImplementedError("Subclase debe implementar get_position")

    def cancel_order(self, order_id: str) -> bool:
        """Cancela una orden pendiente. True si se cancelo."""
        raise NotImplementedError("Subclase debe implementar cancel_order")

    def get_account(self) -> dict:
        """
        Devuelve dict con:
          cash: float
          equity: float
          buying_power: float
          positions: list[Position]
        """
        raise NotImplementedError("Subclase debe implementar get_account")
