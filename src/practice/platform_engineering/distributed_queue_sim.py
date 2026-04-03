"""Practice simulation for distributed queue behavior.

Concepts covered:
- Partitioned logs
- Consumer groups
- Rebalance events
- Retry and dead-letter handling
- Basic lag metrics
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
import random
import time
from typing import Deque, Dict, List


@dataclass
class Message:
    key: str
    payload: str
    attempts: int = 0


@dataclass
class Partition:
    id: int
    queue: Deque[Message] = field(default_factory=deque)

    def produce(self, message: Message) -> None:
        self.queue.append(message)

    def consume(self) -> Message | None:
        if self.queue:
            return self.queue.popleft()
        return None


class Broker:
    def __init__(self, partition_count: int) -> None:
        self.partitions: List[Partition] = [Partition(id=i) for i in range(partition_count)]

    def route(self, key: str) -> Partition:
        index = hash(key) % len(self.partitions)
        return self.partitions[index]

    def produce(self, key: str, payload: str) -> None:
        self.route(key).produce(Message(key=key, payload=payload))


class ConsumerGroup:
    def __init__(self, broker: Broker, consumer_count: int, max_retries: int = 2) -> None:
        self.broker = broker
        self.consumer_count = consumer_count
        self.max_retries = max_retries
        self.dead_letter: List[Message] = []
        self.processed = 0
        self.failed = 0
        self.ownership: Dict[int, int] = {}
        self.rebalance()

    def rebalance(self) -> None:
        self.ownership.clear()
        for p in self.broker.partitions:
            self.ownership[p.id] = p.id % self.consumer_count

    def process(self, message: Message) -> bool:
        # Simulate intermittent failures (~15%).
        return random.random() > 0.15

    def poll_once(self) -> None:
        for partition in self.broker.partitions:
            msg = partition.consume()
            if msg is None:
                continue

            ok = self.process(msg)
            if ok:
                self.processed += 1
                continue

            msg.attempts += 1
            self.failed += 1
            if msg.attempts > self.max_retries:
                self.dead_letter.append(msg)
            else:
                partition.produce(msg)

    def lag(self) -> Dict[int, int]:
        return {p.id: len(p.queue) for p in self.broker.partitions}


def demo() -> None:
    random.seed(7)
    broker = Broker(partition_count=6)

    for i in range(150):
        tenant = f"tenant-{i % 5}"
        broker.produce(key=tenant, payload=f"event-{i}")

    group = ConsumerGroup(broker=broker, consumer_count=3, max_retries=3)

    for tick in range(40):
        if tick == 15:
            group.consumer_count = 4
            group.rebalance()

        group.poll_once()
        current_lag = sum(group.lag().values())
        if tick % 8 == 0:
            print(
                f"tick={tick:02d} lag={current_lag:03d} "
                f"processed={group.processed:03d} failed={group.failed:03d} "
                f"dlq={len(group.dead_letter):02d}"
            )
        if current_lag == 0:
            break
        time.sleep(0.01)

    print("\nFinal metrics")
    print(f"processed={group.processed}")
    print(f"failed={group.failed}")
    print(f"dead_letter={len(group.dead_letter)}")
    lag_by_partition = group.lag()
    print(f"remaining_lag={sum(lag_by_partition.values())}")
    print(f"lag_by_partition={dict(sorted(lag_by_partition.items()))}")

    by_tenant = defaultdict(int)
    for msg in group.dead_letter:
        by_tenant[msg.key] += 1
    if by_tenant:
        print(f"dlq_by_tenant={dict(sorted(by_tenant.items()))}")


if __name__ == "__main__":
    demo()
