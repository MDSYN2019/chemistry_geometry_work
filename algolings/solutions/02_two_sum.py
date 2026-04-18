def two_sum(nums: list[int], target: int) -> tuple[int, int]:
    seen: dict[int, int] = {}
    for i, value in enumerate(nums):
        need = target - value
        if need in seen:
            return (seen[need], i)
        seen[value] = i
    return (-1, -1)
