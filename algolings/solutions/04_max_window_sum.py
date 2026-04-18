def max_window_sum(nums: list[int], k: int) -> int:
    if k <= 0 or k > len(nums):
        return 0
    current = sum(nums[:k])
    best = current
    for i in range(k, len(nums)):
        current += nums[i] - nums[i - k]
        best = max(best, current)
    return best
