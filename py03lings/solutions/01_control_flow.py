"""Solution 01: classify numbers with control flow."""


def classify_numbers(values: list[int]) -> dict[str, int]:
    counts = {"positive": 0, "negative": 0, "zero": 0}
    for value in values:
        if value > 0:
            counts["positive"] += 1
        elif value < 0:
            counts["negative"] += 1
        else:
            counts["zero"] += 1
    return counts


if __name__ == "__main__":
    print(classify_numbers([3, -1, 0, 7, 0, -9]))
