"""Exercise 73: structure a real ablation + failure-analysis workflow.

Deliverables:
- 3-5 ablation experiments
- failure slice report
- concise markdown summary
"""

from dataclasses import dataclass
from pathlib import Path


@dataclass
class AblationResult:
    name: str
    metric: float
    note: str


def rank_runs(results: list[AblationResult]) -> list[AblationResult]:
    """Higher metric is better."""
    return sorted(results, key=lambda r: r.metric, reverse=True)


def write_markdown_report(results: list[AblationResult], out_path: Path) -> None:
    ranked = rank_runs(results)

    lines = [
        "# Ablation Report",
        "",
        "## Ranked runs",
        "",
    ]

    for idx, run in enumerate(ranked, start=1):
        lines.append(f"{idx}. **{run.name}**: metric={run.metric:.4f} — {run.note}")

    lines += [
        "",
        "## Failure analysis prompts",
        "",
        "- Which data slices degrade most?",
        "- What hypotheses explain these failures?",
        "- What follow-up experiments would de-risk deployment?",
    ]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines))


if __name__ == "__main__":
    demo = [
        AblationResult("baseline", 0.702, "reference"),
        AblationResult("+edge_features", 0.731, "improves aromatic systems"),
        AblationResult("+geometry", 0.748, "best overall, slower"),
    ]
    write_markdown_report(demo, Path("artifacts/exp73/ablation_report.md"))
    print("exercise 73 report written")
