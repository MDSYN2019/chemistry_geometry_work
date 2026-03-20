"""Exercise 01: classify numbers with control flow."""


def classify_numbers(values: list[int]) -> dict[str, int]:
    """Return counts for positive, negative, and zero."""
    counts = {"positive": 0, "negative": 0, "zero": 0}
    for value in values:
        # TODO: increment exactly one bucket based on sign
        pass
    return counts


if __name__ == "__main__":
    print(classify_numbers([3, -1, 0, 7, 0, -9]))
