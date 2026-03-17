"""Solution 20."""
from functools import total_ordering


@total_ordering
class Temperature:
    def __init__(self, celsius: float) -> None:
        self.celsius = celsius

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Temperature):
            return NotImplemented
        return self.celsius == other.celsius

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, Temperature):
            return NotImplemented
        return self.celsius < other.celsius
