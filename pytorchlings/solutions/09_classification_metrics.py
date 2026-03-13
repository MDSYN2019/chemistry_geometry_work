import torch


def accuracy(pred: torch.Tensor, target: torch.Tensor) -> float:
    labels = pred.argmax(dim=1)
    return float((labels == target).float().mean().item())


def macro_f1(pred: torch.Tensor, target: torch.Tensor, n_classes: int) -> float:
    labels = pred.argmax(dim=1)
    f1_scores = []
    for c in range(n_classes):
        tp = ((labels == c) & (target == c)).sum().item()
        fp = ((labels == c) & (target != c)).sum().item()
        fn = ((labels != c) & (target == c)).sum().item()
        denom = (2 * tp + fp + fn)
        f1_scores.append((2 * tp / denom) if denom > 0 else 0.0)
    return float(sum(f1_scores) / n_classes)
