"""Exercise 20: refactor fragile sushi dictionaries into first-class objects.

Day 1 focus:
- decide when a named class earns its keep over a dict
- write constructors with instance state
- distinguish class attributes from instance attributes
- avoid the mutable class attribute trap
- make debugging pleasant with useful ``__repr__`` output
"""


class SushiPlate:
    default_currency = "GBP"
    # TODO: keep all mutable per-plate state on instances, not on the class.
    # A shared class-level list would make every plate appear to have the same tags.

    def __init__(self, name: str, color: str, price: float) -> None:
        # TODO: store name, color, and price on this instance.
        # TODO: create a fresh empty tags list for this instance.
        pass

    def add_tag(self, tag: str) -> None:
        # TODO: append tag to this plate's tags.
        pass

    def __repr__(self) -> str:
        # TODO: return a developer-friendly representation such as:
        # SushiPlate(name='salmon nigiri', color='red', price=4.5)
        return "SushiPlate()"


def plate_from_dict(raw_plate: dict[str, object]) -> SushiPlate:
    """Convert a loose restaurant dictionary into a named object."""
    # TODO: read name, color, and price from raw_plate and return a SushiPlate.
    return SushiPlate("TODO", "TODO", 0.0)


if __name__ == "__main__":
    plate = plate_from_dict({"name": "salmon nigiri", "color": "red", "price": 4.5})
    plate.add_tag("popular")
    other = SushiPlate("cucumber maki", "green", 3.0)
    print(plate)
    print(other.tags)  # should still be []
