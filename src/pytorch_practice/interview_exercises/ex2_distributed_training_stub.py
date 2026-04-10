"""Exercise 2: Distributed training stub with PyTorch DDP.

Run example:
  torchrun --nproc_per_node=2 src/pytorch_practice/interview_exercises/ex2_distributed_training_stub.py

Task:
- Fill in DDP initialization, DistributedSampler wiring, and rank-aware logging/checkpointing.
"""

from __future__ import annotations

import os

import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.distributed import DistributedSampler


def setup_ddp() -> tuple[int, int, int]:
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    if world_size > 1:
        dist.init_process_group(backend="gloo")
    return local_rank, rank, world_size


def cleanup_ddp(world_size: int) -> None:
    if world_size > 1 and dist.is_initialized():
        dist.destroy_process_group()


def build_loader(rank: int, world_size: int) -> DataLoader:
    x = torch.randn(2048, 32)
    y = (x[:, :4].sum(dim=1, keepdim=True) > 0).float()
    ds = TensorDataset(x, y)

    sampler = DistributedSampler(ds, num_replicas=world_size, rank=rank, shuffle=True) if world_size > 1 else None
    return DataLoader(ds, batch_size=64, sampler=sampler, shuffle=sampler is None)


def main() -> None:
    local_rank, rank, world_size = setup_ddp()

    device = torch.device("cpu")
    model = nn.Sequential(nn.Linear(32, 64), nn.ReLU(), nn.Linear(64, 1)).to(device)
    if world_size > 1:
        model = DDP(model)

    loader = build_loader(rank, world_size)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.BCEWithLogitsLoss()

    for epoch in range(3):
        if hasattr(loader, "sampler") and isinstance(loader.sampler, DistributedSampler):
            loader.sampler.set_epoch(epoch)

        total = 0.0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = loss_fn(logits, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.item())

        if rank == 0:
            print(f"epoch={epoch} loss={total/len(loader):.4f} world_size={world_size}")

    cleanup_ddp(world_size)


if __name__ == "__main__":
    main()
