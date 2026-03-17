"""Solution 14."""
from functools import cached_property
import math


class Circle:
    def __init__(self, radius: float) -> None:
        self.radius = radius

    @cached_property
    def area(self) -> float:
        return math.pi * self.radius**2
