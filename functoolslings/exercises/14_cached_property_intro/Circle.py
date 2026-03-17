"""Exercise 14: cached_property basics."""
from functools import cached_property
import math


class Circle:
    def __init__(self, radius: float) -> None:
        self.radius = radius

    @cached_property
    def area(self) -> float:
        # TODO: compute pi * r^2
        return 0.0
