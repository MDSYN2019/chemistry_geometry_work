"""Solution 00: build a comma-separated summary with f-strings."""


def summarize_student(name: str, score: int, topics: list[str]) -> str:
    topics_text = ", ".join(topics) if topics else "no topics"
    return f"{name} scored {score} in {topics_text}"


if __name__ == "__main__":
    print(summarize_student("Ada", 93, ["loops", "dicts", "functions"]))
