"""Exercise 06: create and use a dataclass."""

from dataclasses import dataclass, field


@dataclass
class StudentRecord:
    name: str
    scores: list[float] = field(default_factory=list)

    def average(self) -> float:
        # TODO: return 0.0 if there are no scores
        # TODO: otherwise return the arithmetic mean
        return 0.0

    def add_score(self, score: float) -> None:
        # TODO: append score to self.scores
        pass


if __name__ == "__main__":
    record = StudentRecord("Ari")
    record.add_score(88)
    record.add_score(92)
    print(record.average())
