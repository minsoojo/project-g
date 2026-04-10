"""Training and evaluation utilities."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from .metrics import compute_classification_metrics
from .utils import save_json


def build_optimizer(model: nn.Module, lr: float = 1e-4, weight_decay: float = 1e-2) -> torch.optim.Optimizer:
    """Build the AdamW optimizer required by the task."""
    return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    criterion: nn.Module | None = None,
) -> float:
    """Run one training epoch and return the mean loss."""
    model.train()
    criterion = criterion or nn.BCEWithLogitsLoss()
    total_loss = 0.0
    sample_count = 0

    for pixel_values, labels in dataloader:
        pixel_values = pixel_values.to(device)
        labels = labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(pixel_values)
        loss = criterion(logits, labels)
        if torch.isnan(loss):
            raise ValueError("NaN detected in training loss")
        loss.backward()
        optimizer.step()
        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        sample_count += batch_size

    return total_loss / max(sample_count, 1)


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    criterion: nn.Module | None = None,
) -> dict[str, float]:
    """Evaluate the model and return output-spec-compliant metrics."""
    model.eval()
    criterion = criterion or nn.BCEWithLogitsLoss()
    total_loss = 0.0
    sample_count = 0
    all_logits: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []

    for pixel_values, labels in dataloader:
        pixel_values = pixel_values.to(device)
        labels = labels.to(device)
        logits = model(pixel_values)
        loss = criterion(logits, labels)
        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        sample_count += batch_size
        all_logits.append(logits.detach().cpu())
        all_labels.append(labels.detach().cpu())

    logits = torch.cat(all_logits) if all_logits else torch.empty(0)
    labels = torch.cat(all_labels) if all_labels else torch.empty(0)
    metrics = compute_classification_metrics(logits, labels) if len(logits) else {
        "accuracy": 0.0,
        "f1_score": 0.0,
        "roc_auc": 0.0,
    }
    metrics["val_loss"] = total_loss / max(sample_count, 1)
    return metrics


def make_epoch_summary(epoch: int, train_loss: float, val_metrics: dict[str, float]) -> dict[str, float | int]:
    """Create the exact training output JSON contract."""
    return {
        "epoch": epoch,
        "train_loss": float(train_loss),
        "val_loss": float(val_metrics["val_loss"]),
        "accuracy": float(val_metrics["accuracy"]),
        "f1_score": float(val_metrics["f1_score"]),
    }


def save_checkpoint(model: nn.Module, path: str | Path) -> Path:
    """Save model weights to a .pt file."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), target)
    return target


def save_epoch_summary(path: str | Path, summary: dict[str, float | int]) -> Path:
    """Persist training results to JSON."""
    return save_json(path, summary)
