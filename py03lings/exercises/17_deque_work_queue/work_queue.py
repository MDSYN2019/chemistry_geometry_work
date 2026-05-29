"""Exercise 17: choose deque when queue operations happen at both ends.

Lists are excellent dynamic arrays. A deque is the right default for FIFO
queues, bounded recent-history buffers, and round-robin rotation because
``popleft`` and appends at either end are O(1).
"""

from collections import deque

def drain_fifo(events: list[str]) -> list[str]:
    """Process events in arrival order without repeatedly popping index 0."""
    # TODO: load events into a deque, then repeatedly popleft until empty.
    processed: list[str] = []
    return processed


def last_n_events(events: list[str], n: int) -> list[str]:
    """Return the most recent n events using a bounded deque."""
    # TODO: use deque(maxlen=n) so old entries fall out automatically.
    return []


def round_robin(workers: list[str], steps: int) -> list[str]:
    """Return the worker selected at each step, rotating fairly after each pick."""
    # TODO: use a deque; pick the left worker, append/rotate it to the back.
    schedule: list[str] = []
    return schedule


def queue_type_for(operation: str) -> str:
    """Explain whether list or deque is the better fit for common operations."""
    # TODO: return "deque" for fifo, lifo_with_left_end, and round_robin.
    # TODO: return "list" for random_indexing and append_only_scan.
    # TODO: raise ValueError for unknown operations.
    return "list"


if __name__ == "__main__":
    print(drain_fifo(["connect", "read", "write"]))
    print(last_n_events(["a", "b", "c", "d"], 2))
    print(round_robin(["cpu0", "cpu1", "cpu2"], 5))
