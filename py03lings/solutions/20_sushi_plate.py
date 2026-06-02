"""Solution 20: refactor fragile sushi dictionaries into first-class objects."""


class SushiPlate:
    default_currency = "GBP"

    def __init__(self, name: str, color: str, price: float) -> None:
        self.name = name
        self.color = color
        self.price = float(price)
        self.tags: list[str] = []

    def add_tag(self, tag: str) -> None:
        self.tags.append(tag)

    def __repr__(self) -> str:
        return (
            f"SushiPlate(name={self.name!r}, color={self.color!r}, "
            f"price={self.price!r})"
        )


def plate_from_dict(raw_plate: dict[str, object]) -> SushiPlate:
    return SushiPlate(
        name=str(raw_plate["name"]),
        color=str(raw_plate["color"]),
        price=float(raw_plate["price"]),
    )


if __name__ == "__main__":
    plate = plate_from_dict({"name": "salmon nigiri", "color": "red", "price": 4.5})
    plate.add_tag("popular")
    other = SushiPlate("cucumber maki", "green", 3.0)
    print(plate)
    print(other.tags)
