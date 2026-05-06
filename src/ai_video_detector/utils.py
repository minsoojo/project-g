"""Utility helpers for reproducible experiments and JSON output."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Union

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Set random seeds for reproducible CPU/GPU execution."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def ensure_parent(path: Union[str, Path]) -> Path:
    """Create parent directories for the target path."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    return target


def save_json(path: Union[str, Path], payload: Any) -> Path:
    """Write a JSON payload with a stable key order."""
    target = ensure_parent(path)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return target


def log_message(level: str, message: str) -> str:
    """Format messages to the required log spec."""
    normalized = level.upper()
    return f"[{normalized}] {message}"
