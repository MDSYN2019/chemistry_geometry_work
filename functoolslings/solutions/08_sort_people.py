"""Solution 08."""
from functools import cmp_to_key


def compare_people(a: tuple[str, int], b: tuple[str, int]) -> int:
    if a[1] != b[1]:
        return a[1] - b[1]
    return -1 if a[0] < b[0] else 1 if a[0] > b[0] else 0


def sort_people(people: list[tuple[str, int]]) -> list[tuple[str, int]]:
    return sorted(people, key=cmp_to_key(compare_people))
