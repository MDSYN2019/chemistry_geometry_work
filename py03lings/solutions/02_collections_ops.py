"""Solution 02: convert rows into a lookup table."""


def build_price_lookup(rows: list[tuple[str, float]]) -> dict[str, float]:
    lookup: dict[str, float] = {}
    for item, price in rows:
        lookup[item] = price
    return lookup


if __name__ == "__main__":
    data = [("NaCl", 10.5), ("H2O", 1.0), ("NaCl", 11.0)]
    print(build_price_lookup(data))
