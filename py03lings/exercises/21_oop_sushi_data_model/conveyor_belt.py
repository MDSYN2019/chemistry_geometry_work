"""Exercise 21: make custom restaurant objects feel native to Python.

Day 2 focus:
- represent objects with ``__repr__`` and ``__str__``
- make a custom container iterable, indexable, sizeable, and searchable
- use comparison dunders so sorting works without a parallel API
- use ``@classmethod`` as an alternate constructor
"""

from __future__ import annotations


class MenuItem:
    def __init__(self, name: str, price: float) -> None:
        self.name = name
        self.price = float(price)

    @classmethod
    def from_csv(cls, row: str) -> "MenuItem":
        # TODO: parse a "name,price" row and return an item.
        return cls("TODO", 0.0)

    def __repr__(self) -> str:
        # TODO: return MenuItem(name='...', price=...)
        return "MenuItem()"

    def __str__(self) -> str:
        # TODO: return a customer-friendly label like "tuna roll (£5.00)".
        return self.name

    def __eq__(self, other: object) -> bool:
        # TODO: compare by name and price; return NotImplemented for other types.
        return False

    def __lt__(self, other: "MenuItem") -> bool:
        # TODO: sort cheaper items first, using name as the tie-breaker.
        return False


class ConveyorBelt:
    def __init__(self, items: list[MenuItem] | None = None) -> None:
        self._items = list(items or [])

    def add(self, item: MenuItem) -> None:
        self._items.append(item)

    def __len__(self) -> int:
        # TODO: make len(belt) work.
        return 0

    def __iter__(self):
        # TODO: make for item in belt work.
        return iter(())

    def __getitem__(self, index: int) -> MenuItem:
        # TODO: make belt[index] work.
        raise IndexError(index)

    def __contains__(self, name: object) -> bool:
        # TODO: make "tuna roll" in belt check item names.
        return False

    def __add__(self, other: "ConveyorBelt") -> "ConveyorBelt":
        # TODO: return a new belt containing items from both belts.
        return ConveyorBelt()


if __name__ == "__main__":
    belt = ConveyorBelt([MenuItem.from_csv("tuna roll,5"), MenuItem.from_csv("edamame,2.5")])
    print(len(belt), belt[0], "tuna roll" in belt)
    print(sorted(belt))
