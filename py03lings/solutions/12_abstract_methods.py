"""Solution 12: use abstract methods for polymorphism."""

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
        return self.width * self.height


class Circle(Shape):
    def __init__(self, radius: float) -> None:
        self.radius = radius

    def area(self) -> float:
        return 3.14159 * self.radius * self.radius


def total_paint_cost(shapes: list[Shape], price_per_unit: float) -> float:
    return sum(shape.area() for shape in shapes) * price_per_unit


if __name__ == "__main__":
    shapes = [Rectangle(3, 4), Circle(1)]
    print(total_paint_cost(shapes, 2.5))
