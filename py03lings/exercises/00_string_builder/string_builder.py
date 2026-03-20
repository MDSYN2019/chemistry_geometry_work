"""Exercise 00: build a comma-separated summary with f-strings."""


def summarize_student(name: str, score: int, topics: list[str]) -> str:
    # TODO: return exactly: "{name} scored {score} in {topic1, topic2, ...}"
    # TODO: if topics is empty, use "no topics"
    topics_text = ""
    return f"{name} scored {score} in {topics_text}"


if __name__ == "__main__":
    print(summarize_student("Ada", 93, ["loops", "dicts", "functions"]))
