"""Exercise 13: combine mixins with super() for reusable behavior."""


class TimestampMixin:
    def stamp(self) -> str:
        # TODO: return a fixed timestamp string "[2026-01-01T00:00:00Z]"
        return ""


class Message:
    def __init__(self, text: str) -> None:
        self.text = text

    def render(self) -> str:
        return self.text


class StampedMessage(TimestampMixin, Message):
    def render(self) -> str:
        # TODO: call self.stamp()
        # TODO: call super().render()
        # TODO: return "<stamp> <message>"
        return ""


if __name__ == "__main__":
    msg = StampedMessage("Deployment succeeded")
    print(msg.render())
