"""Exercise 19: simulate a simple round-robin CPU scheduler.

Real kernels account for priorities, blocking, wakeups, CPU affinity, and many
hardware details. This deliberately small model focuses on a core idea:
runnable threads get a time slice, unfinished work goes to the back of the run
queue, and completion time depends on the chosen quantum.
"""

from collections import deque
from dataclasses import dataclass


@dataclass(frozen=True)
class ThreadSpec:
    name: str
    cpu_time_ms: int


@dataclass(frozen=True)
class TimeSlice:
    thread: str
    start_ms: int
    end_ms: int


def validate_thread(thread: ThreadSpec) -> None:
    # TODO: raise ValueError when name is empty or cpu_time_ms is less than 1.
    return None


def simulate_round_robin(threads: list[ThreadSpec], quantum_ms: int) -> list[TimeSlice]:
    """Return the execution slices produced by a round-robin scheduler."""
    # TODO: reject quantum_ms < 1.
    # TODO: validate all threads.
    # TODO: keep remaining CPU time in a deque of (ThreadSpec, remaining_ms).
    # TODO: each slice runs min(quantum_ms, remaining_ms).
    # TODO: append unfinished threads to the back of the queue.
    return []


def completion_times(slices: list[TimeSlice]) -> dict[str, int]:
    """Return the final end time for each thread."""
    # TODO: the last slice for a thread determines its completion time.
    return {}


def average_turnaround_ms(threads: list[ThreadSpec], quantum_ms: int) -> float:
    """Assume all threads arrive at time 0 and return average completion time."""
    # TODO: simulate, compute completion times, and average them.
    return 0.0


if __name__ == "__main__":
    workload = [ThreadSpec("render", 5), ThreadSpec("io", 2), ThreadSpec("train", 7)]
    print(simulate_round_robin(workload, quantum_ms=3))
    print(average_turnaround_ms(workload, quantum_ms=3))
