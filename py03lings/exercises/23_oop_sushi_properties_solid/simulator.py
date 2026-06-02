"""Exercise 23: finish a lasting restaurant model with properties and SOLID.

Day 4 focus:
- use ``@property`` for read-only/computed values and validated setters
- use ``@dataclass`` for small value objects
- use ``field(default_factory=...)`` for mutable defaults
- depend on a protocol so pricing behavior is swappable
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol


@dataclass(frozen=True)
class OrderLine:
    item_name: str
    quantity: int
    unit_price: float

    @property
    def subtotal(self) -> float:
        # TODO: compute quantity * unit_price.
        return 0.0


class PricingPolicy(Protocol):
    def price_for(self, base_price: float, hour: int) -> float:
        """Return the effective price for this hour."""


class StandardPricing:
    def price_for(self, base_price: float, hour: int) -> float:
        # TODO: standard pricing does not change the price.
        return 0.0


class HappyHourPricing:
    def price_for(self, base_price: float, hour: int) -> float:
        # TODO: give a 25% discount for orders from 16:00 through 17:59.
        return 0.0


@dataclass
class OrderQueue:
    lines: list[OrderLine] = field(default_factory=list)

    def add(self, line: OrderLine) -> None:
        # TODO: append a line to this queue.
        pass

    @property
    def total(self) -> float:
        # TODO: compute the total of all order lines.
        return 0.0


class RestaurantClock:
    def __init__(self, hour: int) -> None:
        self.hour = hour

    @property
    def hour(self) -> int:
        return self._hour

    @hour.setter
    def hour(self, value: int) -> None:
        # TODO: accept only integers from 0 through 23, then store on _hour.
        self._hour = value


if __name__ == "__main__":
    clock = RestaurantClock(16)
    policy: PricingPolicy = HappyHourPricing()
    queue = OrderQueue()
    queue.add(OrderLine("salmon nigiri", 2, policy.price_for(4.5, clock.hour)))
    print(queue.total)
