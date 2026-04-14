"""Exercise 12: use abstract methods for polymorphism."""

from abc import ABC, abstractmethod


class Shape(ABC):
    @abstractmethod
    def area(self) -> float:
        """Return shape area."""


class Rectangle(Shape):
    def __init__(self, width: float, height: float) -> None:
        self.width = width
        self.height = height

    def area(self) -> float:
        # TODO: return rectangle area
        return 0.0


class Circle(Shape):
    def __init__(self, radius: float) -> None:
        self.radius = radius

    def area(self) -> float:
        # TODO: return circle area using 3.14159 for pi
        return 0.0


def total_paint_cost(shapes: list[Shape], price_per_unit: float) -> float:
    # TODO: sum all areas
    # TODO: multiply by price_per_unit
    return 0.0


if __name__ == "__main__":
    shapes = [Rectangle(3, 4), Circle(1)]
    print(total_paint_cost(shapes, 2.5))
