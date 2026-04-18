import bisect


def lis_length(nums: list[int]) -> int:
    tails: list[int] = []
    for x in nums:
        idx = bisect.bisect_left(tails, x)
        if idx == len(tails):
            tails.append(x)
        else:
            tails[idx] = x
    return len(tails)
