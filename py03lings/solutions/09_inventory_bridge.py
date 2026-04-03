"""Solution 09: combine dataclass, staticmethod, and pydantic."""

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
        return InventoryItem(
            sku=payload.sku,
            quantity=payload.quantity,
            unit_price=payload.unit_price,
        )

    def value(self) -> float:
        return self.quantity * self.unit_price


if __name__ == "__main__":
    incoming = ItemIn(sku="ABC-123", quantity=5, unit_price=9.99)
    item = InventoryItem.from_input(incoming)
    print(item.value())
