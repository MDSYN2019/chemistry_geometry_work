"""Exercise 08: Sort records using cmp_to_key."""
from functools import cmp_to_key


def compare_people(a: tuple[str, int], b: tuple[str, int]) -> int:
    # TODO: sort by age ascending, then name ascending
    return 0


def sort_people(people: list[tuple[str, int]]) -> list[tuple[str, int]]:
    return sorted(people, key=cmp_to_key(compare_people))
