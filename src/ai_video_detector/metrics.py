"""Metrics used by the baseline classifier."""

from __future__ import annotations

import math

import torch


def _safe_divide(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def binary_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    probs = torch.sigmoid(logits)
    preds = (probs >= 0.5).to(labels.dtype)
    return _safe_divide((preds == labels).sum().item(), labels.numel())


def binary_f1_score(logits: torch.Tensor, labels: torch.Tensor) -> float:
    probs = torch.sigmoid(logits)
    preds = (probs >= 0.5).to(labels.dtype)
    true_positive = ((preds == 1) & (labels == 1)).sum().item()
    false_positive = ((preds == 1) & (labels == 0)).sum().item()
    false_negative = ((preds == 0) & (labels == 1)).sum().item()
    precision = _safe_divide(true_positive, true_positive + false_positive)
    recall = _safe_divide(true_positive, true_positive + false_negative)
    return _safe_divide(2 * precision * recall, precision + recall)


def binary_roc_auc(logits: torch.Tensor, labels: torch.Tensor) -> float:
    probs = torch.sigmoid(logits).detach().cpu()
    labels_cpu = labels.detach().cpu().to(torch.int64)
    positives = int((labels_cpu == 1).sum().item())
    negatives = int((labels_cpu == 0).sum().item())
    if positives == 0 or negatives == 0:
        return 0.0

    sorted_indices = torch.argsort(probs, descending=False)
    sorted_probs = probs[sorted_indices]
    sorted_labels = labels_cpu[sorted_indices]

    rank_sum = 0.0
    current_rank = 1
    index = 0
    while index < len(sorted_probs):
        next_index = index + 1
        while next_index < len(sorted_probs) and math.isclose(
            float(sorted_probs[next_index]),
            float(sorted_probs[index]),
            rel_tol=1e-9,
            abs_tol=1e-9,
        ):
            next_index += 1
        average_rank = (current_rank + (current_rank + (next_index - index) - 1)) / 2.0
        positive_count = int((sorted_labels[index:next_index] == 1).sum().item())
        rank_sum += average_rank * positive_count
        current_rank += next_index - index
        index = next_index

    auc = (rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)
    return float(auc)


def compute_classification_metrics(logits: torch.Tensor, labels: torch.Tensor) -> dict[str, float]:
    """Return the metrics required by the task spec."""
    return {
        "accuracy": binary_accuracy(logits, labels),
        "f1_score": binary_f1_score(logits, labels),
        "roc_auc": binary_roc_auc(logits, labels),
    }
