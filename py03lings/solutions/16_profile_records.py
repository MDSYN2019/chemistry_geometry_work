"""Solution 16: use __slots__ when many tiny Python objects need less overhead."""


class PacketSample:
    __slots__ = ("sensor_id", "sequence", "payload_size_bytes")

    def __init__(self, sensor_id: str, sequence: int, payload_size_bytes: int) -> None:
        self.sensor_id = sensor_id
        self.sequence = sequence
        self.payload_size_bytes = payload_size_bytes

    def estimated_wire_bytes(self) -> int:
        return self.payload_size_bytes + 16


def has_per_instance_dict(obj: object) -> bool:
    return hasattr(obj, "__dict__")


def summarize(samples: list[PacketSample]) -> dict[str, int | bool]:
    return {
        "count": len(samples),
        "total_wire_bytes": sum(sample.estimated_wire_bytes() for sample in samples),
        "uses_instance_dict": any(has_per_instance_dict(sample) for sample in samples),
    }


if __name__ == "__main__":
    batch = [PacketSample("temperature", 1, 128), PacketSample("pressure", 2, 64)]
    print(summarize(batch))
