"""Exercise 16: use __slots__ when many tiny Python objects need less overhead.

The goal is not to make every class slotted. Use slots when instances are
numerous, attribute names are fixed, and you do not need a per-instance
``__dict__`` for dynamic attributes.
"""


class PacketSample:
    # TODO: add __slots__ for sensor_id, sequence, and payload_size_bytes.

    def __init__(self, sensor_id: str, sequence: int, payload_size_bytes: int) -> None:
        # TODO: assign the three constructor arguments to instance attributes.
        pass

    def estimated_wire_bytes(self) -> int:
        """Return the payload plus a tiny fixed header estimate."""
        # TODO: each packet has a 16-byte header in addition to the payload.
        return 0


def has_per_instance_dict(obj: object) -> bool:
    """Return True when obj still has a normal per-instance __dict__."""
    # TODO: slotted instances without "__dict__" in __slots__ should return False.
    return True


def summarize(samples: list[PacketSample]) -> dict[str, int | bool]:
    """Summarize a batch without adding dynamic attributes to each sample."""
    # TODO: return count, total_wire_bytes, and uses_instance_dict.
    return {"count": 0, "total_wire_bytes": 0, "uses_instance_dict": True}


if __name__ == "__main__":
    batch = [PacketSample("temperature", 1, 128), PacketSample("pressure", 2, 64)]
    print(summarize(batch))
