"""Solution 21: make custom restaurant objects feel native to Python."""

from __future__ import annotations


class MenuItem:
    def __init__(self, name: str, price: float) -> None:
        self.name = name
        self.price = float(price)

    @classmethod
    def from_csv(cls, row: str) -> "MenuItem":
        name, raw_price = row.split(",", maxsplit=1)
        return cls(name.strip(), float(raw_price))

    def __repr__(self) -> str:
        return f"MenuItem(name={self.name!r}, price={self.price!r})"

    def __str__(self) -> str:
        return f"{self.name} (£{self.price:.2f})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, MenuItem):
            return NotImplemented
        return (self.name, self.price) == (other.name, other.price)

    def __lt__(self, other: "MenuItem") -> bool:
        return (self.price, self.name) < (other.price, other.name)


class ConveyorBelt:
    def __init__(self, items: list[MenuItem] | None = None) -> None:
        self._items = list(items or [])

    def add(self, item: MenuItem) -> None:
        self._items.append(item)

    def __len__(self) -> int:
        return len(self._items)

    def __iter__(self):
        return iter(self._items)

    def __getitem__(self, index: int) -> MenuItem:
        return self._items[index]

    def __contains__(self, name: object) -> bool:
        if not isinstance(name, str):
            return False
        return any(item.name == name for item in self._items)

    def __add__(self, other: "ConveyorBelt") -> "ConveyorBelt":
        return ConveyorBelt([*self._items, *other._items])


if __name__ == "__main__":
    belt = ConveyorBelt([MenuItem.from_csv("tuna roll,5"), MenuItem.from_csv("edamame,2.5")])
    print(len(belt), belt[0], "tuna roll" in belt)
    print(sorted(belt))
