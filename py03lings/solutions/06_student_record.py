"""Solution 06: create and use a dataclass."""

from dataclasses import dataclass, field


@dataclass
class StudentRecord:
    name: str
    scores: list[float] = field(default_factory=list)

    def average(self) -> float:
        if not self.scores:
            return 0.0
        return sum(self.scores) / len(self.scores)

    def add_score(self, score: float) -> None:
        self.scores.append(score)


if __name__ == "__main__":
    record = StudentRecord("Ari")
    record.add_score(88)
    record.add_score(92)
    print(record.average())
