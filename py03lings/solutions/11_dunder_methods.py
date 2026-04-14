"""Solution 11: implement advanced object methods (dunder methods)."""


class Vector2D:
    def __init__(self, x: float, y: float) -> None:
        self.x = x
        self.y = y

    def __repr__(self) -> str:
        return f"Vector2D(x={self.x}, y={self.y})"

    def __add__(self, other: "Vector2D") -> "Vector2D":
        return Vector2D(self.x + other.x, self.y + other.y)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Vector2D):
            return False
        return self.x == other.x and self.y == other.y


if __name__ == "__main__":
    v1 = Vector2D(1, 2)
    v2 = Vector2D(3, 4)
    print(v1 + v2)
    print(v1 == Vector2D(1, 2))
