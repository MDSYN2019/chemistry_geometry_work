"""Exercise 11: implement advanced object methods (dunder methods)."""


class Vector2D:
    def __init__(self, x: float, y: float) -> None:
        self.x = x
        self.y = y

    def __repr__(self) -> str:
        # TODO: return exactly Vector2D(x=<x>, y=<y>)
        return "Vector2D(x=0, y=0)"

    def __add__(self, other: "Vector2D") -> "Vector2D":
        # TODO: return a new vector that adds matching coordinates
        return Vector2D(0, 0)

    def __eq__(self, other: object) -> bool:
        # TODO: return False when other is not Vector2D
        # TODO: compare x and y values for equality
        return False


if __name__ == "__main__":
    v1 = Vector2D(1, 2)
    v2 = Vector2D(3, 4)
    print(v1 + v2)
    print(v1 == Vector2D(1, 2))
