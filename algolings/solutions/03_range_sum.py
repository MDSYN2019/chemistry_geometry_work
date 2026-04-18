def build_prefix(nums: list[int]) -> list[int]:
    prefix = [0]
    for n in nums:
        prefix.append(prefix[-1] + n)
    return prefix


def range_sum(prefix: list[int], left: int, right: int) -> int:
    return prefix[right + 1] - prefix[left]
