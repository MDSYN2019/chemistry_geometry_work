"""Exercise 02: convert rows into a lookup table."""


def build_price_lookup(rows: list[tuple[str, float]]) -> dict[str, float]:
    """Keep the latest price for each item name."""
    lookup: dict[str, float] = {}
    # TODO: fill lookup from rows
    # hint: later rows should overwrite earlier rows with the same key
    return lookup


if __name__ == "__main__":
    data = [("NaCl", 10.5), ("H2O", 1.0), ("NaCl", 11.0)]
    print(build_price_lookup(data))
