"""Solution 23: finish a lasting restaurant model with properties and SOLID."""

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
        return self.quantity * self.unit_price


class PricingPolicy(Protocol):
    def price_for(self, base_price: float, hour: int) -> float:
        """Return the effective price for this hour."""


class StandardPricing:
    def price_for(self, base_price: float, hour: int) -> float:
        return base_price


class HappyHourPricing:
    def price_for(self, base_price: float, hour: int) -> float:
        if 16 <= hour < 18:
            return base_price * 0.75
        return base_price


@dataclass
class OrderQueue:
    lines: list[OrderLine] = field(default_factory=list)

    def add(self, line: OrderLine) -> None:
        self.lines.append(line)

    @property
    def total(self) -> float:
        return sum(line.subtotal for line in self.lines)


class RestaurantClock:
    def __init__(self, hour: int) -> None:
        self.hour = hour

    @property
    def hour(self) -> int:
        return self._hour

    @hour.setter
    def hour(self, value: int) -> None:
        if not isinstance(value, int):
            raise TypeError("hour must be an integer")
        if not 0 <= value <= 23:
            raise ValueError("hour must be between 0 and 23")
        self._hour = value


if __name__ == "__main__":
    clock = RestaurantClock(16)
    policy: PricingPolicy = HappyHourPricing()
    queue = OrderQueue()
    queue.add(OrderLine("salmon nigiri", 2, policy.price_for(4.5, clock.hour)))
    print(queue.total)
