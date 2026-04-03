"""Exercise 09: combine dataclass, staticmethod, and pydantic."""

from dataclasses import dataclass

from pydantic import BaseModel, Field


class ItemIn(BaseModel):
    sku: str
    quantity: int = Field(ge=0)
    unit_price: float = Field(gt=0)


@dataclass
class InventoryItem:
    sku: str
    quantity: int
    unit_price: float

    @staticmethod
    def from_input(payload: ItemIn) -> "InventoryItem":
        # TODO: build and return InventoryItem from payload fields
        return InventoryItem(sku="", quantity=0, unit_price=1.0)

    def value(self) -> float:
        # TODO: return quantity * unit_price
        return 0.0


if __name__ == "__main__":
    incoming = ItemIn(sku="ABC-123", quantity=5, unit_price=9.99)
    item = InventoryItem.from_input(incoming)
    print(item.value())
