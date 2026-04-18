"""Exercise 02: two-sum with hashing.

TODO:
- Return indices (i, j) where nums[i] + nums[j] == target.
- Return (-1, -1) if no pair exists.
"""


def two_sum(nums: list[int], target: int) -> tuple[int, int]:
    # TODO: use hash map for O(n)
    raise NotImplementedError


if __name__ == "__main__":
    assert two_sum([2, 7, 11, 15], 9) == (0, 1)
    assert two_sum([1, 2, 3], 10) == (-1, -1)
    print("ok")
