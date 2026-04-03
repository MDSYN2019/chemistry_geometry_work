"""Solution 08: validate data with a Pydantic model."""

from pydantic import BaseModel, Field


class Order(BaseModel):
    item: str
    quantity: int = Field(gt=0)
    unit_price: float = Field(gt=0)

    def total(self) -> float:
        return self.quantity * self.unit_price


if __name__ == "__main__":
    order = Order(item="notebook", quantity=3, unit_price=2.5)
    print(order.total())
