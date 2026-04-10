"""Exercise 72: mini research-engineering training framework.

Checklist:
- deterministic seeds
- config-driven setup
- checkpointing
- metric logging
- multi-GPU readiness hooks
"""

from dataclasses import dataclass, asdict
import json
from pathlib import Path
import random

import torch


@dataclass
class ExperimentConfig:
    seed: int = 42
    run_name: str = "baseline"
    lr: float = 1e-3
    batch_size: int = 64
    epochs: int = 3
    output_dir: str = "artifacts/exp72"


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def save_config(cfg: ExperimentConfig) -> Path:
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = out_dir / "config.json"
    cfg_path.write_text(json.dumps(asdict(cfg), indent=2))
    return cfg_path


def save_checkpoint(path: Path, model: torch.nn.Module, optimizer: torch.optim.Optimizer, step: int) -> None:
    payload = {
        "step": step,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(payload, path)


def train_stub(cfg: ExperimentConfig) -> dict[str, float]:
    set_seed(cfg.seed)
    save_config(cfg)

    model = torch.nn.Linear(10, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)

    last_loss = 0.0
    for step in range(cfg.epochs):
        x = torch.randn(cfg.batch_size, 10)
        y = torch.randint(0, 2, (cfg.batch_size,))

        optimizer.zero_grad()
        logits = model(x)
        loss = torch.nn.functional.cross_entropy(logits, y)
        loss.backward()
        optimizer.step()
        last_loss = float(loss.item())

    ckpt_path = Path(cfg.output_dir) / "last.ckpt"
    save_checkpoint(ckpt_path, model, optimizer, cfg.epochs)

    # TODO: add profiler traces and distributed training hooks
    return {"final_loss": last_loss}


if __name__ == "__main__":
    metrics = train_stub(ExperimentConfig())
    assert "final_loss" in metrics
    print(metrics)
