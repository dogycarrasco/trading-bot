"""Base broker interface para ejecución paper/live.
Implementado por: BinancePaperBroker, IBKRPaperBroker, AlpacaPaperBroker."""
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from typing import Optional


@dataclass
class Order:
    symbol: str
    side: str  # "buy" | "sell"
    qty: float
    order_type: str = "market"  # "market" | "limit" | "stop"
    price: Optional[float] = None
    status: str = "pending"  # "pending" | "filled" | "cancelled" | "rejected"
    order_id: Optional[str] = None


@dataclass
class Position:
    symbol: str
    qty: float
    avg_entry_price: float
    current_price: float = 0.0
    unrealized_pnl: float = 0.0


class BaseBroker(ABC):
    """Interfaz base para todos los brokers."""
    name: str = "base"

    @abstractmethod
    def place_order(self, order: Order) -> Order:
        """Envía una orden y retorna el resultado."""
        pass

    @abstractmethod
    def get_position(self, symbol: str) -> Optional[Position]:
        """Retorna la posición actual para un símbolo."""
        pass

    @abstractmethod
    def cancel_order(self, order_id: str) -> bool:
        """Cancela una orden pendiente."""
        pass

    @abstractmethod
    def get_account(self) -> dict:
        """Retorna info de la cuenta: cash, equity, buying_power."""
        pass
