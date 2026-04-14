"""Solution 13: combine mixins with super() for reusable behavior."""


class TimestampMixin:
    def stamp(self) -> str:
        return "[2026-01-01T00:00:00Z]"


class Message:
    def __init__(self, text: str) -> None:
        self.text = text

    def render(self) -> str:
        return self.text


class StampedMessage(TimestampMixin, Message):
    def render(self) -> str:
        return f"{self.stamp()} {super().render()}"


if __name__ == "__main__":
    msg = StampedMessage("Deployment succeeded")
    print(msg.render())
