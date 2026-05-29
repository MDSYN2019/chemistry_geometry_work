"""Solution 19: simulate a simple round-robin CPU scheduler."""

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
    if not thread.name:
        raise ValueError("thread name must be non-empty")
    if thread.cpu_time_ms < 1:
        raise ValueError("cpu_time_ms must be at least 1")


def simulate_round_robin(threads: list[ThreadSpec], quantum_ms: int) -> list[TimeSlice]:
    if quantum_ms < 1:
        raise ValueError("quantum_ms must be at least 1")
    for thread in threads:
        validate_thread(thread)

    runnable = deque((thread, thread.cpu_time_ms) for thread in threads)
    now_ms = 0
    slices: list[TimeSlice] = []

    while runnable:
        thread, remaining_ms = runnable.popleft()
        run_ms = min(quantum_ms, remaining_ms)
        start_ms = now_ms
        now_ms += run_ms
        slices.append(TimeSlice(thread=thread.name, start_ms=start_ms, end_ms=now_ms))

        remaining_ms -= run_ms
        if remaining_ms > 0:
            runnable.append((thread, remaining_ms))

    return slices


def completion_times(slices: list[TimeSlice]) -> dict[str, int]:
    completed: dict[str, int] = {}
    for time_slice in slices:
        completed[time_slice.thread] = time_slice.end_ms
    return completed


def average_turnaround_ms(threads: list[ThreadSpec], quantum_ms: int) -> float:
    if not threads:
        return 0.0
    completed = completion_times(simulate_round_robin(threads, quantum_ms))
    return sum(completed[thread.name] for thread in threads) / len(threads)


if __name__ == "__main__":
    workload = [ThreadSpec("render", 5), ThreadSpec("io", 2), ThreadSpec("train", 7)]
    print(simulate_round_robin(workload, quantum_ms=3))
    print(average_turnaround_ms(workload, quantum_ms=3))
