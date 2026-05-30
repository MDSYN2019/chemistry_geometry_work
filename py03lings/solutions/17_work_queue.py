"""Solution 17: choose deque when queue operations happen at both ends."""

from collections import deque


def drain_fifo(events: list[str]) -> list[str]:
    queue = deque(events)
    processed: list[str] = []
    while queue:
        processed.append(queue.popleft())
    return processed


def last_n_events(events: list[str], n: int) -> list[str]:
    return list(deque(events, maxlen=n))


def round_robin(workers: list[str], steps: int) -> list[str]:
    if not workers or steps <= 0:
        return []
    queue = deque(workers)
    schedule: list[str] = []
    for _ in range(steps):
        worker = queue.popleft()
        schedule.append(worker)
        queue.append(worker)
    return schedule


def queue_type_for(operation: str) -> str:
    deque_ops = {"fifo", "lifo_with_left_end", "round_robin"}
    list_ops = {"random_indexing", "append_only_scan"}
    if operation in deque_ops:
        return "deque"
    if operation in list_ops:
        return "list"
    raise ValueError(f"unknown operation: {operation}")


if __name__ == "__main__":
    print(drain_fifo(["connect", "read", "write"]))
    print(last_n_events(["a", "b", "c", "d"], 2))
    print(round_robin(["cpu0", "cpu1", "cpu2"], 5))
