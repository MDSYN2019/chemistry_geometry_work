"""Exercise 01: binary search on a sorted list.

TODO:
- Implement iterative binary search.
- Return index of target or -1 if absent.
"""


def binary_search(nums: list[int], target: int) -> int:
    # TODO: use left/right pointers
    raise NotImplementedError


if __name__ == "__main__":
    assert binary_search([1, 3, 5, 8, 13], 8) == 3
    assert binary_search([1, 3, 5, 8, 13], 2) == -1
    print("ok")
