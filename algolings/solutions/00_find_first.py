def find_first(nums: list[int], target: int) -> int:
    for i, value in enumerate(nums):
        if value == target:
            return i
    return -1
